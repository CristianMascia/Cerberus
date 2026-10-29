"""
Cerberus — batch API (`cerberus batch ...`).

Runs a JSONL file of chat requests against the models of a models.conf, with
repetitions and a split into parallel shards, each served by its own llama.cpp
servers:

    validate  check the JSONL against the config (labels, custom_id, params)
    submit    (login node) expand repeats, split into shards, compute the
              placement once, write manifest + shards + job.sbatch, `sbatch --array`
    run       (inside the allocation, one per array task) start the servers,
              process one shard, append results line by line, stop the servers
    status    progress per shard (+ SLURM state when available)
    merge     join the shard results into results.jsonl, in input order

Input, one request per line (`model` and `params` optional):

    {"custom_id": "q-1", "model": "qwen", "messages": [...], "params": {"temperature": 0.2}}

The OpenAI Batch format ({"custom_id", "method", "url", "body": {...}}) is accepted
too and normalised to the form above.

Batch dir layout:

    manifest.json   settings, placement needs, SLURM job id
    requests.jsonl  normalised copy of the input
    shards/shard_XXX.jsonl   the work units (custom_id, repeat) of each shard
    results/shard_XXX.jsonl  one line per completed unit (appended + flushed)
    logs/           SLURM output of the array tasks
    job.sbatch      the array job
    results.jsonl   written by `merge`
"""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace as _replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config, load_config

# Request params: standard OpenAI arguments, llama.cpp extras (sent in extra_body)
# and the reasoning switches (sent as chat_template_kwargs).
OPENAI_PARAMS = ("temperature", "top_p", "max_tokens", "seed", "stop",
                 "presence_penalty", "frequency_penalty")
LLAMACPP_PARAMS = ("top_k", "min_p", "repeat_penalty")
REASONING_PARAMS = ("reasoning", "reasoning_effort")
ALLOWED_PARAMS = OPENAI_PARAMS + LLAMACPP_PARAMS + REASONING_PARAMS
# OpenAI Batch body keys that are not params.
_BODY_SKIP = ("model", "messages", "stream")

CHARS_PER_TOKEN = 3  # pessimistic estimate used to flag prompts that may not fit


class BatchError(Exception):
    """Raised for any problem in the batch input or the batch dir."""


@dataclass
class BatchRequest:
    custom_id: str
    model: str
    messages: list
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #
def _normalise(obj: dict, where: str) -> dict:
    """Turn an OpenAI Batch line into {custom_id, model, messages, params}."""
    if "body" not in obj:
        return obj
    body = obj["body"]
    if not isinstance(body, dict):
        raise BatchError(f"{where}: 'body' must be an object")
    url = obj.get("url", "/v1/chat/completions")
    if url != "/v1/chat/completions":
        raise BatchError(f"{where}: only url '/v1/chat/completions' is supported, got {url!r}")
    params = {k: v for k, v in body.items() if k not in _BODY_SKIP}
    if "max_completion_tokens" in params:
        params.setdefault("max_tokens", params.pop("max_completion_tokens"))
    out = {"custom_id": obj.get("custom_id"), "messages": body.get("messages"),
           "params": params}
    if "model" in body:
        out["model"] = body["model"]
    return out


def load_jsonl(path: str | Path, default_model: Optional[str] = None) -> list[BatchRequest]:
    """Parse the input JSONL. Raises BatchError naming the offending line."""
    path = Path(path)
    if not path.is_file():
        raise BatchError(f"input file not found: {path}")
    requests: list[BatchRequest] = []
    seen: set[str] = set()
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchError(f"{where}: invalid JSON — {exc}") from exc
            if not isinstance(obj, dict):
                raise BatchError(f"{where}: each line must be a JSON object")
            obj = _normalise(obj, where)

            cid = obj.get("custom_id")
            if not isinstance(cid, str) or not cid:
                raise BatchError(f"{where}: 'custom_id' must be a non-empty string")
            if cid in seen:
                raise BatchError(f"{where}: duplicate custom_id '{cid}'")
            seen.add(cid)
            where = f"{where} ({cid})"

            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise BatchError(f"{where}: 'messages' must be a non-empty list")
            for m in msgs:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise BatchError(f"{where}: every message needs 'role' and 'content'")

            model = obj.get("model") or default_model
            if not model:
                raise BatchError(f"{where}: no 'model' on the line and no --model given")

            params = obj.get("params") or {}
            if not isinstance(params, dict):
                raise BatchError(f"{where}: 'params' must be an object")
            unknown = sorted(set(params) - set(ALLOWED_PARAMS))
            if unknown:
                raise BatchError(f"{where}: unknown params {unknown} "
                                 f"(allowed: {list(ALLOWED_PARAMS)})")
            requests.append(BatchRequest(cid, model, msgs, params))
    if not requests:
        raise BatchError(f"{path}: no requests found")
    return requests


def _prompt_chars(messages: list) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):  # OpenAI content parts
            total += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return total


def check_against_config(requests: list[BatchRequest], cfg: Config) -> list[str]:
    """Check labels and limits. Raises BatchError on errors, returns warnings."""
    labels = {m.label for m in cfg.models}
    warnings = []
    for r in requests:
        if r.model not in labels:
            raise BatchError(f"{r.custom_id}: unknown model '{r.model}' "
                             f"(models.conf has {sorted(labels)})")
        spec = cfg.by_label(r.model)
        mt = r.params.get("max_tokens")
        if mt is not None:
            if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
                raise BatchError(f"{r.custom_id}: max_tokens must be an integer >= 1")
            if mt > spec.max_output_tokens:
                raise BatchError(f"{r.custom_id}: max_tokens={mt} exceeds max_output_tokens="
                                 f"{spec.max_output_tokens} of '{spec.label}'")
        est = _prompt_chars(r.messages) // CHARS_PER_TOKEN
        if est > spec.max_input_tokens:
            warnings.append(f"{r.custom_id}: prompt may exceed max_input_tokens="
                            f"{spec.max_input_tokens} of '{spec.label}' (~{est} tokens)")
    return warnings


def subset_config(cfg: Config, labels) -> Config:
    """The config restricted to the models the batch uses (others are not started)."""
    keep = set(labels)
    return _replace(cfg, models=[m for m in cfg.models if m.label in keep])


# --------------------------------------------------------------------------- #
# Repeats and shards
# --------------------------------------------------------------------------- #
def expand_units(requests: list[BatchRequest], repeats: int) -> list[tuple[str, int]]:
    """Work units (custom_id, repeat), repeats expanded before the split.

    Request-major order, so the round-robin split sends the repeats of a request
    to different shards (independent samples, balanced load).
    """
    return [(r.custom_id, k) for r in requests for k in range(repeats)]


def make_shards(units: list, n_shards: int) -> list[list]:
    """Round-robin split; deterministic, never more shards than units."""
    n = max(1, min(n_shards, len(units)))
    return [units[i::n] for i in range(n)]


def _shard_name(i: int) -> str:
    return f"shard_{i:03d}.jsonl"


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a killed task: the unit is redone
    return rows


# --------------------------------------------------------------------------- #
# Prepare / submit
# --------------------------------------------------------------------------- #
def default_batch_dir(cfg: Config) -> Path:
    base = cfg.path.parent if cfg.path else Path.cwd()
    return base / f"batch_{datetime.now():%Y%m%d_%H%M%S}"


def prepare(cfg: Config, input_path: str | Path, batch_dir: Path, *,
            model: Optional[str] = None, shards: Optional[int] = None,
            repeats: Optional[int] = None, estimate: bool = True,
            token: Optional[str] = None) -> dict:
    """Validate, split and write the batch dir. Returns the manifest."""
    shards = shards if shards is not None else cfg.batch.shards
    repeats = repeats if repeats is not None else cfg.batch.repeats
    if shards < 1 or repeats < 1:
        raise BatchError("--shards and --repeats must be >= 1")

    requests = load_jsonl(input_path, default_model=model)
    for w in check_against_config(requests, cfg):
        print(f"[cerberus] warning: {w}")
    labels = sorted({r.model for r in requests}, key=[m.label for m in cfg.models].index)
    sub = subset_config(cfg, labels)

    units = expand_units(requests, repeats)
    split = make_shards(units, shards)
    if len(split) < shards:
        print(f"[cerberus] only {len(units)} unit(s): using {len(split)} shard(s)")
    per_shard = len(units) / len(split)
    if len(split) > 1 and per_shard < 10:
        print(f"[cerberus] warning: ~{per_shard:.0f} request(s) per shard; every shard "
              "pays the server start-up, consider fewer --shards")

    needs, n_nodes = None, None
    if estimate:
        from . import placement
        print("[cerberus] computing GPU/VRAM needs (reads model metadata over HTTP)...")
        need_objs = placement.compute_needs(sub, token=token)
        placements, n_nodes = placement.plan(need_objs, sub.gpus_per_node, sub.vram_gib)
        needs = [{"label": n.spec.label, "num_gpus": n.num_gpus,
                  "footprint_gib": n.footprint_gib} for n in need_objs]
        for p in placements:
            print(f"  {p.spec.label}: num_gpus={p.num_gpus} ~{p.footprint_gib:.1f} GiB "
                  f"ctx={p.spec.ctx_size}")
        print(f"[cerberus] each shard needs {n_nodes} node(s) x {sub.gpus_per_node} GPU(s)")

    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise BatchError(f"batch dir {batch_dir} already exists and is not empty")
    for d in ("shards", "results", "logs"):
        (batch_dir / d).mkdir(parents=True, exist_ok=True)

    _write_jsonl(batch_dir / "requests.jsonl",
                 [{"custom_id": r.custom_id, "model": r.model, "messages": r.messages,
                   "params": r.params} for r in requests])
    for i, us in enumerate(split):
        _write_jsonl(batch_dir / "shards" / _shard_name(i),
                     [{"custom_id": c, "repeat": k} for c, k in us])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(cfg.path).resolve()) if cfg.path else None,
        "input": str(Path(input_path).resolve()),
        "models": labels,
        "n_requests": len(requests),
        "repeats": repeats,
        "shards": len(split),
        "n_units": len(units),
        "gpus_per_node": sub.gpus_per_node,
        "nodes_per_shard": n_nodes,
        "needs": needs,
        "slurm": {
            "partition": cfg.batch.partition, "account": cfg.batch.account,
            "time": cfg.batch.time, "cpus_per_task": cfg.batch.cpus_per_task,
            "max_parallel": cfg.batch.max_parallel,
        },
        "job_id": None,
    }
    _save_manifest(batch_dir, manifest)
    print(f"[cerberus] batch dir: {batch_dir}  ({len(requests)} request(s) x "
          f"{repeats} repeat(s) = {len(units)} unit(s) in {len(split)} shard(s))")
    return manifest


def _save_manifest(batch_dir: Path, manifest: dict) -> None:
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(batch_dir: str | Path) -> dict:
    path = Path(batch_dir) / "manifest.json"
    if not path.is_file():
        raise BatchError(f"no manifest.json in {batch_dir} — not a batch dir?")
    return json.loads(path.read_text())


def write_sbatch(batch_dir: Path, manifest: dict, python: Optional[str] = None) -> Path:
    """Write job.sbatch: one array task per shard, each running `batch run`."""
    sl = manifest["slurm"]
    n = manifest["shards"]
    array = f"0-{n - 1}" + (f"%{sl['max_parallel']}" if sl.get("max_parallel") else "")
    nodes = manifest.get("nodes_per_shard") or 1
    lines = [
        "#!/usr/bin/env bash",
        "#",
        "# Cerberus batch — SLURM job array, one task per shard (written by",
        "# `cerberus batch submit`). Every task starts its own llama.cpp servers.",
        "#",
        f"#SBATCH --job-name=cerberus-batch",
        f"#SBATCH --array={array}",
        f"#SBATCH --nodes={nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={manifest['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={sl['cpus_per_task']}",
        f"#SBATCH --time={sl['time']}",
        f"#SBATCH --output={batch_dir / 'logs' / 'shard_%a_%A.out'}",
    ]
    if sl.get("partition"):
        lines.append(f"#SBATCH --partition={sl['partition']}")
    if sl.get("account"):
        lines.append(f"#SBATCH --account={sl['account']}")
    lines += [
        "",
        "set -euo pipefail",
        f'exec "{python or sys.executable}" -m cerberus batch run '
        f'--batch-dir "{batch_dir}" --shard "$SLURM_ARRAY_TASK_ID"',
        "",
    ]
    path = batch_dir / "job.sbatch"
    path.write_text("\n".join(lines))
    return path


def submit(batch_dir: Path, manifest: dict) -> str:
    """sbatch the array job and record its id in the manifest."""
    if not shutil.which("sbatch"):
        raise BatchError("sbatch not found — run 'cerberus batch submit' on the login node")
    (batch_dir / "logs").mkdir(exist_ok=True)  # SLURM does not create the --output dir
    out = subprocess.run(["sbatch", "--parsable", str(batch_dir / "job.sbatch")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise BatchError(f"sbatch failed: {out.stderr.strip()}")
    job_id = out.stdout.strip().split(";")[0]
    manifest["job_id"] = job_id
    _save_manifest(batch_dir, manifest)
    return job_id


# --------------------------------------------------------------------------- #
# Run one shard
# --------------------------------------------------------------------------- #
def _load_requests(batch_dir: Path) -> dict[str, dict]:
    return {r["custom_id"]: r for r in _read_jsonl(batch_dir / "requests.jsonl")}


def _request_kwargs(params: dict, repeat: int, max_output_tokens: int) -> dict:
    """Map the request params onto CerberusClient.achat() kwargs."""
    p = dict(params)
    kw: dict = {"max_tokens": p.pop("max_tokens", max_output_tokens)}
    if "temperature" in p:
        kw["temperature"] = p.pop("temperature")
    if "seed" in p:  # distinct but reproducible samples across repeats
        kw["seed"] = p.pop("seed") + repeat
    if "reasoning" in p:
        kw["reasoning"] = bool(p.pop("reasoning"))
    extra_body: dict = {}
    if "reasoning_effort" in p:
        extra_body.setdefault("chat_template_kwargs", {})["reasoning_effort"] = \
            p.pop("reasoning_effort")
    for k in LLAMACPP_PARAMS:
        if k in p:
            extra_body[k] = p.pop(k)
    kw.update(p)  # remaining OpenAI params (top_p, stop, penalties)
    if extra_body:
        kw["extra_body"] = extra_body
    return kw


def _is_transient(exc: BaseException) -> bool:
    try:
        import openai
    except ModuleNotFoundError:  # pragma: no cover
        return False
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                        openai.RateLimitError, openai.InternalServerError)):
        return True
    return isinstance(exc, (ConnectionError, asyncio.TimeoutError))


async def _process(client, units: list[tuple[str, int]], requests: dict, cfg: Config,
                   shard: int, out_fh, max_retries: int, timeout: float) -> tuple[int, int]:
    sems = {m.label: asyncio.Semaphore(m.parallel) for m in cfg.models}
    done = errors = 0
    total = len(units)

    async def one(cid: str, repeat: int):
        nonlocal done, errors
        req = requests[cid]
        spec = cfg.by_label(req["model"])
        kw = _request_kwargs(req.get("params") or {}, repeat, spec.max_output_tokens)
        row = {"custom_id": cid, "repeat": repeat, "model": spec.label,
               "response": None, "error": None, "shard": shard, "latency_s": None}
        async with sems[spec.label]:
            t0 = time.monotonic()
            for attempt in range(max_retries + 1):
                try:
                    r = await client.achat(spec.label, req["messages"], timeout=timeout, **kw)
                    usage = getattr(r.raw, "usage", None)
                    row["response"] = {
                        "content": r.content, "reasoning": r.reasoning,
                        "finish_reason": r.finish_reason,
                        "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                  "completion_tokens": getattr(usage, "completion_tokens", None)},
                    }
                    row["error"] = None
                    break
                except Exception as exc:  # recorded per request, never stops the shard
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                    if attempt < max_retries and _is_transient(exc):
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    break
            row["latency_s"] = round(time.monotonic() - t0, 3)
        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_fh.flush()
        done += 1
        errors += row["error"] is not None
        if done % 10 == 0 or done == total:
            print(f"[cerberus] shard {shard}: {done}/{total} done, {errors} error(s)",
                  flush=True)

    await asyncio.gather(*(one(c, k) for c, k in units))
    return done, errors


def run_shard(batch_dir: str | Path, shard: int, *, endpoints: Optional[str] = None,
              retry_failed: bool = False, project_dir: Optional[Path] = None) -> int:
    """Process one shard. Returns the number of units that ended with an error.

    Without `endpoints`, starts the shard's own servers with `deploy.start()` on the
    current SLURM allocation, using the placement needs computed at submit time.
    """
    from .client import CerberusClient

    batch_dir = Path(batch_dir).resolve()
    manifest = load_manifest(batch_dir)
    if not 0 <= shard < manifest["shards"]:
        raise BatchError(f"shard {shard} out of range (batch has {manifest['shards']})")
    cfg = subset_config(load_config(manifest["config"]), manifest["models"])
    requests = _load_requests(batch_dir)
    units = [(u["custom_id"], u["repeat"])
             for u in _read_jsonl(batch_dir / "shards" / _shard_name(shard))]

    out_path = batch_dir / "results" / _shard_name(shard)
    previous = {}
    for row in _read_jsonl(out_path):
        previous[(row["custom_id"], row["repeat"])] = row  # last line wins
    skip = {k for k, row in previous.items() if row.get("error") is None or not retry_failed}
    todo = [u for u in units if u not in skip]
    print(f"[cerberus] shard {shard}: {len(units)} unit(s), {len(units) - len(todo)} "
          f"already done, {len(todo)} to run")
    if not todo:
        return sum(1 for u in units if previous.get(u, {}).get("error") is not None)

    dep = None
    if endpoints:
        client = CerberusClient(endpoints_path=endpoints)
        missing = set(manifest["models"]) - set(client.list_models())
        if missing:
            raise BatchError(f"{endpoints} has no endpoint for {sorted(missing)}")
    else:
        from . import deploy, placement
        needs = None
        if manifest.get("needs"):
            needs = [placement.ModelNeed(cfg.by_label(n["label"]), n["num_gpus"],
                                         n["footprint_gib"]) for n in manifest["needs"]]
        # SIGTERM (scancel, time limit) -> SystemExit, so the finally stops the servers
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
        dep = deploy.start(cfg, project_dir or batch_dir, needs=needs)
        ep_path = deploy.write_endpoints(dep, dep.rundir / "endpoints.json")
        client = CerberusClient(endpoints_path=str(ep_path))

    try:
        with out_path.open("a") as fh:
            _, errors = asyncio.run(_process(
                client, todo, requests, cfg, shard, fh,
                cfg.batch.max_retries, cfg.batch.request_timeout))
    finally:
        if dep is not None:
            from . import deploy
            deploy.stop(dep)

    final = dict(previous)
    for row in _read_jsonl(out_path):
        final[(row["custom_id"], row["repeat"])] = row
    n_err = sum(1 for u in units if final.get(u, {}).get("error") is not None)
    print(f"[cerberus] shard {shard} finished: {len(units)} unit(s), {n_err} with error "
          f"-> {out_path}")
    return n_err


# --------------------------------------------------------------------------- #
# Status / merge
# --------------------------------------------------------------------------- #
def shard_progress(batch_dir: Path, manifest: dict) -> list[dict]:
    rows = []
    for i in range(manifest["shards"]):
        units = {(u["custom_id"], u["repeat"])
                 for u in _read_jsonl(batch_dir / "shards" / _shard_name(i))}
        last = {}
        for r in _read_jsonl(batch_dir / "results" / _shard_name(i)):
            last[(r["custom_id"], r["repeat"])] = r
        done = [k for k in units if k in last]
        errors = sum(1 for k in done if last[k].get("error") is not None)
        rows.append({"shard": i, "total": len(units), "done": len(done), "errors": errors})
    return rows


def slurm_state(job_id: str) -> dict[int, str]:
    """State per array task, from sacct (or squeue); empty if SLURM is not here."""
    states: dict[int, str] = {}
    if shutil.which("sacct"):
        out = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "JobID,State"],
                             capture_output=True, text=True)
        for line in out.stdout.splitlines():
            jid, _, state = line.partition("|")
            if "_" in jid:
                idx = jid.split("_", 1)[1]
                if idx.isdigit():
                    states[int(idx)] = state.split()[0] if state else "?"
    elif shutil.which("squeue"):
        out = subprocess.run(["squeue", "-j", job_id, "-h", "-r", "-o", "%K|%T"],
                             capture_output=True, text=True)
        for line in out.stdout.splitlines():
            idx, _, state = line.partition("|")
            if idx.isdigit():
                states[int(idx)] = state
    return states


def merge(batch_dir: str | Path, out: Optional[str | Path] = None) -> tuple[Path, list]:
    """Join the shard results into results.jsonl, ordered as the input then by repeat.

    Returns (path, missing units). If a unit appears more than once (a resumed
    shard), the last line wins.
    """
    batch_dir = Path(batch_dir)
    manifest = load_manifest(batch_dir)
    order = [r["custom_id"] for r in _read_jsonl(batch_dir / "requests.jsonl")]
    results = {}
    for i in range(manifest["shards"]):
        for r in _read_jsonl(batch_dir / "results" / _shard_name(i)):
            results[(r["custom_id"], r["repeat"])] = r
    rows, missing = [], []
    for cid in order:
        for k in range(manifest["repeats"]):
            if (cid, k) in results:
                rows.append(results[(cid, k)])
            else:
                missing.append((cid, k))
    out_path = Path(out) if out else batch_dir / "results.jsonl"
    _write_jsonl(out_path, rows)
    return out_path, missing
