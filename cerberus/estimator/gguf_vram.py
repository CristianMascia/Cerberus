#!/usr/bin/env python3
"""
Estimate llama.cpp VRAM usage for a GGUF model hosted on Hugging Face.

Reads the GGUF header metadata directly over HTTP range requests (no full
download), gets the weight size from the file size(s), and computes the
KV cache from your max input/output tokens. Handles multi-part (split) GGUFs.

Examples:
    python gguf_vram.py Qwen/Qwen3-8B-GGUF -q Q4_K_M --input 8192 --output 2048
    python gguf_vram.py unsloth/Qwen3-30B-A3B-GGUF -q Q8_0 --gpus 4 --vram 32
    python gguf_vram.py bartowski/Meta-Llama-3.1-8B-Instruct-GGUF -q Q6_K \
        --input 4096 --output 512 --parallel 10 --cache-type q8_0

Notes:
  * Weights = actual on-disk size of the .gguf file(s): this already reflects
    the chosen quantization, so it is exact for a full GPU offload (-ngl 999).
  * KV cache is exact given the architecture metadata; input+output tokens are
    summed (they share the same cache) and multiplied by --parallel slots.
  * Compute buffer / CUDA context are ESTIMATES. For the real number, read the
    'KV self size' and 'compute buffer size' lines that llama-server prints on
    load. This tool errs on the safe side.
"""

import argparse
import json
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"

# GGUF metadata value-type enum
(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32,
 T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64) = range(13)

_SCALAR = {
    T_U8: ("<B", 1), T_I8: ("<b", 1), T_U16: ("<H", 2), T_I16: ("<h", 2),
    T_U32: ("<I", 4), T_I32: ("<i", 4), T_F32: ("<f", 4), T_BOOL: ("<?", 1),
    T_U64: ("<Q", 8), T_I64: ("<q", 8), T_F64: ("<d", 8),
}

# Metadata suffixes we care about (key is "<arch>.<suffix>")
CARE_SUFFIX = (
    "block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "embedding_length",
    "attention.key_length",
    "attention.value_length",
    "context_length",
)
CARE_EXACT = ("general.architecture", "general.parameter_count")

# Bytes per KV element for common cache quantizations (K and V each)
CACHE_BYTES = {"f16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Streaming GGUF header reader (fetches bytes lazily from a `fetch` callback)
# --------------------------------------------------------------------------- #
class Reader:
    def __init__(self, fetch, chunk=262144):
        self._fetch = fetch
        self._chunk = chunk
        self._buf = bytearray()
        self.cursor = 0

    def _ensure(self, upto):
        while len(self._buf) < upto:
            start = len(self._buf)
            data = self._fetch(start, start + self._chunk)
            if not data:
                raise EOFError("unexpected end of GGUF stream while parsing header")
            self._buf += data

    def read(self, n):
        self._ensure(self.cursor + n)
        b = bytes(self._buf[self.cursor:self.cursor + n])
        self.cursor += n
        return b

    def skip(self, n):
        self._ensure(self.cursor + n)
        self.cursor += n

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")


def _read_value(r, vtype, want):
    """Read (want=True) or skip (want=False) one metadata value."""
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, r.read(size))[0]
    if vtype == T_STR:
        return r.string()
    if vtype == T_ARR:
        et = r.u32()
        cnt = r.u64()
        if want:
            return [_read_value(r, et, True) for _ in range(cnt)]
        # skip as cheaply as possible
        if et in _SCALAR:
            r.skip(cnt * _SCALAR[et][1])
        elif et == T_STR:
            for _ in range(cnt):
                r.skip(r.u64())
        else:
            for _ in range(cnt):
                _read_value(r, et, False)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def _have_core(md):
    arch = md.get("general.architecture")
    if not arch:
        return False
    need = ("block_count", "embedding_length",
            "attention.head_count", "context_length")
    return all(f"{arch}.{s}" in md for s in need)


def parse_gguf_header(fetch):
    r = Reader(fetch)
    if r.read(4) != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    version = r.u32()
    if version < 2:
        raise ValueError(f"unsupported GGUF version {version}")
    _ = r.u64()          # tensor_count (unused here)
    kv_count = r.u64()

    md = {}
    for _ in range(kv_count):
        key = r.string()
        vtype = r.u32()
        want = key in CARE_EXACT or any(key.endswith("." + s) for s in CARE_SUFFIX)
        # once we have the core arch params, stop before the big tokenizer arrays
        if vtype == T_ARR and not want and _have_core(md):
            break
        val = _read_value(r, vtype, want)
        if want:
            md[key] = val
    return md


# --------------------------------------------------------------------------- #
# Hugging Face helpers
# --------------------------------------------------------------------------- #
def _auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_list_gguf(repo, token):
    url = f"{HF}/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"Accept": "application/json", **_auth(token)})
    with urllib.request.urlopen(req) as resp:
        tree = json.load(resp)
    return [(e["path"], e.get("size", 0)) for e in tree
            if e.get("type") == "file" and e["path"].lower().endswith(".gguf")]


def make_range_fetch(url, token):
    hdr = _auth(token)

    def fetch(start, end):
        req = urllib.request.Request(
            url, headers={**hdr, "Range": f"bytes={start}-{end - 1}"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 416:      # requested range past EOF
                return b""
            raise

    return fetch


def split_group(path):
    """Return the common prefix key for a split file, or None if not split."""
    # matches "...-00001-of-00009.gguf"
    marker = path.rfind("-of-")
    if marker == -1:
        return None
    dash = path.rfind("-", 0, marker)
    if dash == -1:
        return None
    return path[:dash]        # everything before "-000NN-of-..."


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def estimate(md, weight_bytes, in_tok, out_tok, parallel, cache_type, gpus, vram_gb):
    arch = md["general.architecture"]

    def g(suffix, default=None):
        return md.get(f"{arch}.{suffix}", default)

    n_layer = g("block_count")
    n_embd = g("embedding_length")
    n_head = g("attention.head_count")
    n_head_kv = g("attention.head_count_kv", n_head)
    if isinstance(n_head_kv, list):          # per-layer GQA: use the max
        n_head_kv = max(n_head_kv)
    train_ctx = g("context_length")
    head_dim = g("attention.key_length") or (n_embd // n_head)
    params = md.get("general.parameter_count")

    seq = in_tok + out_tok
    total_ctx = seq * parallel
    bpe = CACHE_BYTES[cache_type]
    n_embd_kv = n_head_kv * head_dim

    kv_bytes = 2 * n_layer * total_ctx * n_embd_kv * bpe          # 2 = K and V

    # --- estimated (not exact) portions ---
    cuda_ctx = 0.4 * gpus * GIB                                   # per-device context
    compute = (0.3 + (total_ctx / 8192) * 0.2) * GIB             # rough graph/activations
    margin = 0.05 * weight_bytes                                  # safety pad

    total = weight_bytes + kv_bytes + cuda_ctx + compute + margin
    per_gpu = total / gpus if gpus else total

    return {
        "arch": arch, "params": params, "n_layer": n_layer, "n_embd": n_embd,
        "n_head": n_head, "n_head_kv": n_head_kv, "head_dim": head_dim,
        "train_ctx": train_ctx, "seq": seq, "total_ctx": total_ctx,
        "weights": weight_bytes, "kv": kv_bytes, "cuda_ctx": cuda_ctx,
        "compute": compute, "margin": margin, "total": total,
        "per_gpu": per_gpu, "vram_gb": vram_gb, "gpus": gpus,
    }


def estimate_from_hf(repo, gguf_file=None, quant=None, *, in_tok=4096, out_tok=512,
                     parallel=1, cache_type="f16", gpus=1, vram_gb=32, token=None):
    """List the repo, select the target GGUF (exact filename or quant substring),
    read its header over HTTP range requests, and return (estimate_dict, shown_name).

    Importable entry point used by the Cerberus placement engine.
    """
    files = hf_list_gguf(repo, token)
    if not files:
        raise ValueError(f"no .gguf files found in {repo}")
    sizes = dict(files)

    if gguf_file:
        target = gguf_file
        if target not in sizes:
            raise ValueError(f"{target} not found in repo {repo}")
    elif quant:
        matches = [p for p in sizes if quant.lower() in p.lower()]
        if not matches:
            raise ValueError(f"no file matches quant '{quant}' in {repo}")
        target = sorted(matches, key=len)[0]
    elif len(sizes) == 1:
        target = next(iter(sizes))
    else:
        raise ValueError(f"multiple GGUF files in {repo}; specify gguf_file or quant")

    grp = split_group(target)
    if grp:
        parts = sorted(p for p in sizes if split_group(p) == grp)
        weight_bytes = sum(sizes[p] for p in parts)
        meta_file = parts[0]
        shown = f"{grp}-*.gguf ({len(parts)} parts)"
    else:
        weight_bytes = sizes[target]
        meta_file = target
        shown = target

    url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(meta_file)}"
    md = parse_gguf_header(make_range_fetch(url, token))
    est = estimate(md, weight_bytes, in_tok, out_tok, parallel, cache_type, gpus, vram_gb)
    return est, shown


def gb(x):
    return f"{x / GIB:7.2f} GiB"


def report(e, filename, cache_type):
    fit = e["per_gpu"] <= e["vram_gb"] * GIB
    p = e["params"]
    pstr = f"{p/1e9:.1f}B" if p else "n/a"
    lines = [
        f"File            : {filename}",
        f"Architecture    : {e['arch']}   params: {pstr}",
        f"Layers          : {e['n_layer']}   n_embd: {e['n_embd']}   "
        f"heads: {e['n_head']}   kv-heads: {e['n_head_kv']}   head_dim: {e['head_dim']}",
        f"Trained ctx     : {e['train_ctx']}",
        f"Requested ctx   : {e['seq']} tok/seq x {e['total_ctx']//e['seq']} "
        f"slot(s) = {e['total_ctx']} tok   (KV cache type: {cache_type})",
        "-" * 52,
        f"Weights         : {gb(e['weights'])}",
        f"KV cache        : {gb(e['kv'])}",
        f"Compute buffer  : {gb(e['compute'])}   (estimate)",
        f"CUDA context    : {gb(e['cuda_ctx'])}   (estimate, {e['gpus']} GPU)",
        f"Safety margin   : {gb(e['margin'])}",
        "-" * 52,
        f"TOTAL           : {gb(e['total'])}",
        f"Per GPU (/{e['gpus']})    : {gb(e['per_gpu'])}   "
        f"vs {e['vram_gb']:.0f} GiB  ->  {'FITS' if fit else 'DOES NOT FIT'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id, e.g. Qwen/Qwen3-8B-GGUF")
    ap.add_argument("-q", "--quant", help="quant substring to match, e.g. Q4_K_M")
    ap.add_argument("-f", "--file", help="exact .gguf filename (overrides --quant)")
    ap.add_argument("--input", type=int, default=4096, help="max input tokens")
    ap.add_argument("--output", type=int, default=512, help="max output tokens")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent slots (llama.cpp --parallel)")
    ap.add_argument("--cache-type", choices=list(CACHE_BYTES), default="f16",
                    help="KV cache quantization (default f16)")
    ap.add_argument("--gpus", type=int, default=1, help="number of GPUs to split across")
    ap.add_argument("--vram", type=float, default=32, help="VRAM per GPU in GiB")
    ap.add_argument("--token", help="HF token (or set env HF_TOKEN)")
    args = ap.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        est, shown = estimate_from_hf(
            args.repo, gguf_file=args.file, quant=args.quant,
            in_tok=args.input, out_tok=args.output, parallel=args.parallel,
            cache_type=args.cache_type, gpus=args.gpus, vram_gb=args.vram, token=token)
    except urllib.error.HTTPError as e:
        sys.exit(f"error: HTTP {e.code} {e.reason} for repo '{args.repo}'")
    except ValueError as e:
        sys.exit(f"error: {e}")
    except Exception as e:                    # noqa: BLE001
        sys.exit(f"error: failed to estimate: {e}")

    print(report(est, shown, args.cache_type))


if __name__ == "__main__":
    main()
