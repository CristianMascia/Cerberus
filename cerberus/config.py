"""
Cerberus — parsing and validation of the declarative model-serving spec (TOML).

A models.conf declares an [allocation] budget, optional [defaults], one
[[model]] table per model to serve and an optional [batch] section with the
defaults of `cerberus batch`. `load_config()` returns a validated `Config`
with typed `ModelSpec` entries; any problem raises `ConfigError` with a message
that names the offending model/field.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # stdlib on Python >= 3.11; fallback for older interpreters
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ALLOC_MODES = ("AUTO", "MANUAL")
# llamacpp: GGUF served by llama-server; vllm: safetensors served by `vllm serve`.
BACKENDS = ("llamacpp", "vllm")
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
# vLLM --kv-cache-dtype: 'f16' keeps the model dtype (vLLM's 'auto'), 'fp8' halves the KV.
VLLM_KV_CACHE_TYPES = ("f16", "fp8")
# vLLM runs many sequences per GPU: used when the [[model]] table sets no `parallel`
# ([defaults] parallel is a llama.cpp slot count and does not apply to vLLM).
VLLM_DEFAULT_PARALLEL = 64
VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
REASONING_MODES = ("on", "off", "auto")
# llama.cpp --reasoning-format: how the server separates thinking from the answer.
# 'deepseek' (default) parses <think>…</think>; 'auto' lets llama.cpp pick per model
# (needed for harmony models like gpt-oss, whose channels 'deepseek' does not parse);
# 'none' leaves the raw output in content for the client to parse.
REASONING_FORMATS = ("deepseek", "deepseek-legacy", "auto", "none")
# VRAM of one GPU when [allocation] does not say: V100-SXM2-32GB (IBiSCo).
DEFAULT_VRAM_GIB = 32.0


class ConfigError(Exception):
    """Raised for any schema/validation problem in models.conf."""


@dataclass
class ModelSpec:
    label: str
    hf_repo: str
    gguf_file: str                  # llamacpp only ("" for vllm)
    alloc_mode: str                 # AUTO | MANUAL
    max_input_tokens: int
    max_output_tokens: int
    parallel: int = 1
    kv_cache_type: str = "f16"
    reasoning: str = "auto"         # on | off | auto
    reasoning_format: str = "deepseek"  # llama.cpp --reasoning-format (see REASONING_FORMATS)
    reasoning_budget: Optional[int] = None
    num_gpus: Optional[int] = None  # required for MANUAL, ignored for AUTO
    backend: str = "llamacpp"       # llamacpp | vllm
    revision: Optional[str] = None  # vllm: HF branch/tag/commit (default: main)
    gpu_memory_utilization: float = VLLM_DEFAULT_GPU_MEMORY_UTILIZATION  # vllm
    reasoning_parser: Optional[str] = None  # vllm --reasoning-parser (e.g. qwen3)

    @property
    def is_vllm(self) -> bool:
        return self.backend == "vllm"

    @property
    def ctx_size(self) -> int:
        """llama.cpp: total --ctx-size, each of `parallel` slots gets in+out tokens.
        vLLM: --max-model-len, the length of ONE sequence (KV is paged and shared)."""
        per_seq = self.max_input_tokens + self.max_output_tokens
        return per_seq if self.is_vllm else per_seq * self.parallel

    @property
    def is_split_name(self) -> bool:
        """True if gguf_file looks like the first part of a split GGUF."""
        import re
        return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", self.gguf_file))


@dataclass
class BatchSettings:
    """Defaults of `cerberus batch` (the optional [batch] section); CLI flags win."""
    shards: int = 1                 # parallel SLURM array tasks, each with its own servers
    repeats: int = 1                # repetitions of every request
    partition: Optional[str] = None
    account: Optional[str] = None
    time: str = "04:00:00"          # time limit of each array task
    cpus_per_task: int = 8
    max_parallel: Optional[int] = None  # max array tasks running at once (--array=...%N)
    max_retries: int = 3            # per-request retries on transient errors
    request_timeout: float = 600.0  # seconds per request


@dataclass
class Config:
    gpus_per_node: int
    models: list[ModelSpec] = field(default_factory=list)
    path: Optional[Path] = None
    batch: BatchSettings = field(default_factory=BatchSettings)
    vram_gib: float = DEFAULT_VRAM_GIB  # usable VRAM of ONE GPU

    def by_label(self, label: str) -> ModelSpec:
        for m in self.models:
            if m.label == label:
                return m
        raise KeyError(label)


# --------------------------------------------------------------------------- #
def _require(table: dict, key: str, where: str, typ):
    if key not in table:
        raise ConfigError(f"{where}: missing required field '{key}'")
    val = table[key]
    if not isinstance(val, typ) or (typ is int and isinstance(val, bool)):
        raise ConfigError(f"{where}: field '{key}' must be {typ.__name__}, got {val!r}")
    return val


def _model_from_table(t: dict, defaults: dict, idx: int) -> ModelSpec:
    where = f"model[{idx}]"
    label = _require(t, "label", where, str)
    where = f"model '{label}'"

    backend = str(t.get("backend", defaults.get("backend", "llamacpp")))
    if backend not in BACKENDS:
        raise ConfigError(f"{where}: backend must be one of {BACKENDS}")
    vllm = backend == "vllm"

    hf_repo = _require(t, "hf_repo", where, str)
    if vllm:
        if "gguf_file" in t:
            raise ConfigError(f"{where}: 'gguf_file' is for llamacpp; vllm serves the "
                              "safetensors of hf_repo")
        gguf_file = ""
    else:
        gguf_file = _require(t, "gguf_file", where, str)
        for key in ("revision", "gpu_memory_utilization", "reasoning_parser"):
            if key in t:
                raise ConfigError(f"{where}: '{key}' is only for backend = \"vllm\"")

    alloc_mode = _require(t, "alloc_mode", where, str).upper()
    if alloc_mode not in ALLOC_MODES:
        raise ConfigError(f"{where}: alloc_mode must be one of {ALLOC_MODES}")

    max_in = _require(t, "max_input_tokens", where, int)
    max_out = _require(t, "max_output_tokens", where, int)
    if max_in <= 0 or max_out <= 0:
        raise ConfigError(f"{where}: max_input_tokens/max_output_tokens must be > 0")

    if vllm:
        parallel = int(t.get("parallel", VLLM_DEFAULT_PARALLEL))
    else:
        parallel = int(t.get("parallel", defaults.get("parallel", 1)))
    if parallel < 1:
        raise ConfigError(f"{where}: parallel must be >= 1")

    kv = str(t.get("kv_cache_type", defaults.get("kv_cache_type", "f16")))
    allowed_kv = VLLM_KV_CACHE_TYPES if vllm else KV_CACHE_TYPES
    if kv not in allowed_kv:
        raise ConfigError(f"{where}: kv_cache_type must be one of {allowed_kv} "
                          f"for backend {backend}")

    reasoning = str(t.get("reasoning", defaults.get("reasoning", "auto")))
    if reasoning not in REASONING_MODES:
        raise ConfigError(f"{where}: reasoning must be one of {REASONING_MODES}")

    reasoning_format = str(
        t.get("reasoning_format", defaults.get("reasoning_format", "deepseek"))
    )
    if reasoning_format not in REASONING_FORMATS:
        raise ConfigError(f"{where}: reasoning_format must be one of {REASONING_FORMATS}")

    reasoning_budget = t.get("reasoning_budget")
    if reasoning_budget is not None and (not isinstance(reasoning_budget, int)
                                         or isinstance(reasoning_budget, bool)):
        raise ConfigError(f"{where}: reasoning_budget must be an integer")
    if vllm and reasoning_budget is not None:
        raise ConfigError(f"{where}: reasoning_budget is not supported by vllm")

    revision = t.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ConfigError(f"{where}: revision must be a string")
    gmu = t.get("gpu_memory_utilization", VLLM_DEFAULT_GPU_MEMORY_UTILIZATION)
    if isinstance(gmu, bool) or not isinstance(gmu, (int, float)) or not 0 < gmu <= 1:
        raise ConfigError(f"{where}: gpu_memory_utilization must be a number in (0, 1]")
    reasoning_parser = t.get("reasoning_parser")
    if reasoning_parser is not None and not isinstance(reasoning_parser, str):
        raise ConfigError(f"{where}: reasoning_parser must be a string")

    num_gpus = t.get("num_gpus")
    if alloc_mode == "MANUAL":
        if num_gpus is None:
            raise ConfigError(f"{where}: MANUAL alloc_mode requires 'num_gpus'")
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
            raise ConfigError(f"{where}: num_gpus must be an integer >= 1")
    else:  # AUTO
        if num_gpus is not None:
            raise ConfigError(f"{where}: 'num_gpus' is only for MANUAL (AUTO computes it)")

    return ModelSpec(
        label=label, hf_repo=hf_repo, gguf_file=gguf_file, alloc_mode=alloc_mode,
        max_input_tokens=max_in, max_output_tokens=max_out, parallel=parallel,
        kv_cache_type=kv, reasoning=reasoning, reasoning_format=reasoning_format,
        reasoning_budget=reasoning_budget, num_gpus=num_gpus, backend=backend,
        revision=revision, gpu_memory_utilization=float(gmu),
        reasoning_parser=reasoning_parser,
    )


def _batch_from_table(t: dict) -> BatchSettings:
    where = "[batch]"
    known = set(BatchSettings.__dataclass_fields__)
    for key in t:
        if key not in known:
            raise ConfigError(f"{where}: unknown field '{key}' (allowed: {sorted(known)})")
    b = BatchSettings()
    for key in ("shards", "repeats", "cpus_per_task", "max_retries"):
        if key in t:
            val = _require(t, key, where, int)
            if val < (0 if key == "max_retries" else 1):
                raise ConfigError(f"{where}: '{key}' must be >= "
                                  f"{0 if key == 'max_retries' else 1}")
            setattr(b, key, val)
    for key in ("partition", "account", "time"):
        if key in t:
            setattr(b, key, _require(t, key, where, str))
    if "max_parallel" in t:
        val = _require(t, "max_parallel", where, int)
        if val < 1:
            raise ConfigError(f"{where}: 'max_parallel' must be >= 1")
        b.max_parallel = val
    if "request_timeout" in t:
        val = t["request_timeout"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
            raise ConfigError(f"{where}: 'request_timeout' must be a number > 0")
        b.request_timeout = float(val)
    return b


def load_config(path: str | Path) -> Config:
    """Parse and validate a models.conf (TOML). Raises ConfigError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: invalid TOML — {exc}") from exc

    alloc = data.get("allocation", {})
    gpus_per_node = alloc.get("gpus_per_node")
    if not isinstance(gpus_per_node, int) or isinstance(gpus_per_node, bool) or gpus_per_node < 1:
        raise ConfigError("[allocation]: 'gpus_per_node' must be an integer >= 1")
    vram_gib = alloc.get("vram_gib", DEFAULT_VRAM_GIB)
    if isinstance(vram_gib, bool) or not isinstance(vram_gib, (int, float)) or vram_gib <= 0:
        raise ConfigError("[allocation]: 'vram_gib' must be a number > 0 (GiB of one GPU)")

    defaults = data.get("defaults", {})
    raw_models = data.get("model", [])
    if not raw_models:
        raise ConfigError("no [[model]] entries found in the config")

    models: list[ModelSpec] = []
    seen: set[str] = set()
    for i, t in enumerate(raw_models):
        m = _model_from_table(t, defaults, i)
        if m.label in seen:
            raise ConfigError(f"duplicate model label '{m.label}'")
        seen.add(m.label)
        if m.alloc_mode == "MANUAL" and m.num_gpus > gpus_per_node:
            raise ConfigError(
                f"model '{m.label}': num_gpus={m.num_gpus} exceeds gpus_per_node={gpus_per_node} "
                "(tensor-split cannot cross nodes)"
            )
        models.append(m)

    batch = _batch_from_table(data.get("batch", {}))
    return Config(gpus_per_node=gpus_per_node, models=models, path=path, batch=batch,
                  vram_gib=float(vram_gib))


if __name__ == "__main__":  # quick manual check: python -m cerberus.config <file>
    cfg = load_config(sys.argv[1])
    print(f"gpus_per_node={cfg.gpus_per_node}, vram_gib={cfg.vram_gib:g}, "
          f"{len(cfg.models)} models")
    for m in cfg.models:
        print(f"  {m.label}: {m.backend} {m.alloc_mode} ctx={m.ctx_size} reasoning={m.reasoning} "
              f"num_gpus={m.num_gpus}")
