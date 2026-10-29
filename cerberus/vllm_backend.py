"""
Cerberus — what the vLLM backend needs besides launching (`node_launch.sh`).

vLLM serves the safetensors of `hf_repo` and preallocates
`gpu_memory_utilization` x VRAM on every GPU it uses, whatever the model size.
So a vLLM model always takes WHOLE GPUs: the estimate here only decides HOW MANY
(the tensor-parallel size), checking that weights + the KV of one full sequence +
runtime overhead fit in the preallocated share of each GPU.

Network (login node): `model_info()` for the file sizes and `config.json` for the
KV geometry. `required_gpus_for()` is pure so it can be tested offline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import ModelSpec

GIB = 1024 ** 3
# CUDA context, activations, sampler and CUDA graphs per GPU (vLLM, H100-class).
OVERHEAD_GIB = 6.0
# What `cerberus download` skips: other weight formats and duplicate checkpoints.
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.h5", "*.msgpack",
                   "*.ckpt", "original/*", "consolidated*"]
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def weights_bytes(spec: ModelSpec, token: str | None = None) -> int:
    """Total size of the repo's *.safetensors (what vLLM loads), from the HF API."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(spec.hf_repo, revision=spec.revision,
                              files_metadata=True, token=token)
    sizes = [s.size or 0 for s in info.siblings
             if s.rfilename.endswith(".safetensors")
             and not s.rfilename.startswith(("original/", "consolidated"))]
    if not sizes:
        raise RuntimeError(f"model '{spec.label}': no .safetensors in {spec.hf_repo} "
                           "(vllm needs the HF weights, not a GGUF)")
    return sum(sizes)


def model_config(spec: ModelSpec, token: str | None = None) -> dict:
    import json
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec.hf_repo, "config.json", revision=spec.revision, token=token)
    return json.loads(Path(path).read_text())


def kv_geometry(cfg: dict) -> tuple[int, int]:
    """(KV elements per token, attention heads) from a HF config.json.

    Multimodal configs keep the language model under `text_config`.
    """
    c = cfg.get("text_config") or cfg
    heads = int(c["num_attention_heads"])
    kv_heads = int(c.get("num_key_value_heads") or heads)
    head_dim = int(c.get("head_dim") or c["hidden_size"] // heads)
    layers = int(c["num_hidden_layers"])
    return 2 * layers * kv_heads * head_dim, heads  # 2 = K and V


def kv_dtype_bytes(spec: ModelSpec, cfg: dict) -> int:
    if spec.kv_cache_type == "fp8":
        return 1
    c = cfg.get("text_config") or cfg
    dtype = (c.get("dtype") or c.get("torch_dtype")
             or cfg.get("dtype") or cfg.get("torch_dtype"))
    return _DTYPE_BYTES.get(str(dtype), 2)


def required_gpus_for(weights_gib: float, kv_seq_gib: float, heads: int,
                      gpus_per_node: int, vram_gib: float, gpu_memory_utilization: float,
                      label: str = "?") -> int:
    """Smallest tensor-parallel size whose per-GPU share fits.

    Per GPU: weights/g + KV of one full sequence/g + OVERHEAD_GIB must fit in
    vram_gib * gpu_memory_utilization; g must divide the attention heads.
    """
    budget = vram_gib * gpu_memory_utilization
    for g in range(1, gpus_per_node + 1):
        if heads % g:
            continue
        if (weights_gib + kv_seq_gib) / g + OVERHEAD_GIB <= budget:
            return g
    raise RuntimeError(
        f"model '{label}' does not fit on {gpus_per_node} GPU(s): weights "
        f"{weights_gib:.1f} GiB + KV of one sequence {kv_seq_gib:.1f} GiB vs "
        f"{budget:.0f} GiB usable per GPU"
    )


def required_gpus(spec: ModelSpec, gpus_per_node: int, vram_gib: float,
                  token: str | None = None) -> tuple[int, float]:
    """(num_gpus, footprint_gib) for a vllm model. Footprint = the whole GPUs."""
    token = token or os.environ.get("HF_TOKEN")
    if spec.alloc_mode == "MANUAL":
        return spec.num_gpus, vram_gib * spec.num_gpus
    cfg = model_config(spec, token)
    kv_elems, heads = kv_geometry(cfg)
    kv_seq_gib = kv_elems * kv_dtype_bytes(spec, cfg) * spec.ctx_size / GIB
    weights_gib = weights_bytes(spec, token) / GIB
    g = required_gpus_for(weights_gib, kv_seq_gib, heads, gpus_per_node, vram_gib,
                          spec.gpu_memory_utilization, spec.label)
    return g, vram_gib * g


def download(spec: ModelSpec, token: str | None = None) -> str:
    """Fetch the snapshot vLLM needs (safetensors, configs, tokenizer) into the HF cache."""
    from huggingface_hub import snapshot_download
    return snapshot_download(spec.hf_repo, revision=spec.revision, token=token,
                             ignore_patterns=IGNORE_PATTERNS)


def resolve_local_snapshot(spec: ModelSpec, hf_cache: str | Path) -> Path | None:
    """The cached snapshot dir of `spec` (with its config.json), or None."""
    repo = Path(hf_cache) / "hub" / ("models--" + spec.hf_repo.replace("/", "--"))
    rev = spec.revision or "main"
    if _HEX40.match(rev):
        commit = rev
    else:
        ref = repo / "refs" / rev
        if not ref.is_file():
            return None
        commit = ref.read_text().strip()
    snap = repo / "snapshots" / commit
    return snap if (snap / "config.json").exists() else None
