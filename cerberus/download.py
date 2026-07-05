"""
Cerberus — model download.

Fetches, for each model in the config, exactly the declared GGUF file (and, for
split GGUFs, all of its parts) into the shared Hugging Face cache. Runs on the
login node (the only one with external network). Auth via the `HF_TOKEN` set in
the `cerberus` conda env; needed only for gated repos.

`huggingface_hub` is imported lazily so the rest of the package stays importable
without it (and off-cluster).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import Config, ModelSpec

_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _files_to_fetch(spec: ModelSpec) -> list[str]:
    """The GGUF filenames to download for a model (all parts if it is split)."""
    m = _SPLIT_RE.match(spec.gguf_file)
    if not m:
        return [spec.gguf_file]
    base, total = m.group("base"), int(m.group("total"))
    return [f"{base}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def download_models(config: Config, token: str | None = None) -> dict[str, list[str]]:
    """Download every model's GGUF(s). Returns {label: [local_path, ...]}.

    Skips files already present in the cache (huggingface_hub handles caching).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for 'cerberus download' "
            "(pip install -e . in the cerberus env)"
        ) from exc

    token = token or os.environ.get("HF_TOKEN")
    out: dict[str, list[str]] = {}
    for spec in config.models:
        paths = []
        for fname in _files_to_fetch(spec):
            print(f"[cerberus] download {spec.label}: {spec.hf_repo}/{fname}")
            local = hf_hub_download(repo_id=spec.hf_repo, filename=fname, token=token)
            paths.append(local)
        out[spec.label] = paths
    print(f"[cerberus] downloaded {len(out)} model(s) into the HF cache.")
    return out


def resolve_local_gguf(spec: ModelSpec, hf_cache: str | Path) -> Path | None:
    """Find the local path of a model's (first) GGUF in the HF cache, or None.

    Snapshots store files as symlinks into blobs, so we follow them.
    """
    repo_dir = "models--" + spec.hf_repo.replace("/", "--")
    snapshots = Path(hf_cache) / "hub" / repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in snapshots.iterdir():
        cand = snap / spec.gguf_file
        if cand.exists():
            #return cand.resolve()
            return cand
    return None
