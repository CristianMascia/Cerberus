"""
Cerberus — VRAM estimation and GPU-count decision.

Thin, importable wrapper around `vram_estimator/gguf_vram.py`. Given a `ModelSpec`
it returns the model's VRAM footprint (weights + KV-cache + CUDA context + compute
buffer + margin) and, for AUTO models, the smallest number of GPUs it fits on.

Reads GGUF headers over HTTP range requests (no full download); needs network on
the machine that runs it (the login node) and honours `HF_TOKEN`.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

from .config import ModelSpec

_GGUF_VRAM_PATH = Path(__file__).resolve().parent.parent / "vram_estimator" / "gguf_vram.py"


@lru_cache(maxsize=1)
def _gv():
    """Load vram_estimator/gguf_vram.py as a module (sibling dir, not a package)."""
    if not _GGUF_VRAM_PATH.is_file():
        raise RuntimeError(f"vram estimator not found at {_GGUF_VRAM_PATH}")
    spec = importlib.util.spec_from_file_location("gguf_vram", _GGUF_VRAM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def estimate(spec: ModelSpec, gpus: int, vram_gib: float = 32.0, token: str | None = None):
    """Return the estimate dict for `spec` assuming a `gpus`-way split."""
    gv = _gv()
    token = token or os.environ.get("HF_TOKEN")
    est, _shown = gv.estimate_from_hf(
        spec.hf_repo, gguf_file=spec.gguf_file,
        in_tok=spec.max_input_tokens, out_tok=spec.max_output_tokens,
        parallel=spec.parallel, cache_type=spec.kv_cache_type,
        gpus=gpus, vram_gb=vram_gib, token=token,
    )
    return est


def footprint_gib(est) -> float:
    """Total VRAM footprint (GiB) from an estimate dict."""
    return est["total"] / _gv().GIB


def per_gpu_gib(est) -> float:
    return est["per_gpu"] / _gv().GIB


def required_gpus(spec: ModelSpec, gpus_per_node: int, vram_gib: float = 32.0,
                  token: str | None = None) -> tuple[int, dict]:
    """Decide how many GPUs `spec` needs and return (num_gpus, estimate_dict).

    MANUAL: uses spec.num_gpus (estimate computed at that split for packing).
    AUTO: smallest g in 1..gpus_per_node whose per-GPU footprint fits in vram_gib.
          CUDA context grows with g, so this is a genuine fit search, not a divide.
    """
    if spec.alloc_mode == "MANUAL":
        est = estimate(spec, spec.num_gpus, vram_gib, token)
        return spec.num_gpus, est

    last = None
    for g in range(1, gpus_per_node + 1):
        est = estimate(spec, g, vram_gib, token)
        last = est
        if est["per_gpu"] <= vram_gib * _gv().GIB:
            return g, est
    # Did not fit even across a whole node.
    need = footprint_gib(last) if last else float("nan")
    raise RuntimeError(
        f"model '{spec.label}' does not fit on {gpus_per_node} GPU(s) "
        f"(needs ~{need:.1f} GiB total, {per_gpu_gib(last):.1f} GiB/GPU at "
        f"{gpus_per_node}-way split vs {vram_gib:.0f} GiB)"
    )
