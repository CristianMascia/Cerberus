"""
Cerberus — parsing and validation of the declarative model-serving spec (TOML).

A models.conf declares an [allocation] budget, optional [defaults], and one
[[model]] table per model to serve. `load_config()` returns a validated `Config`
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
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
REASONING_MODES = ("on", "off", "auto")


class ConfigError(Exception):
    """Raised for any schema/validation problem in models.conf."""


@dataclass
class ModelSpec:
    label: str
    hf_repo: str
    gguf_file: str
    alloc_mode: str                 # AUTO | MANUAL
    max_input_tokens: int
    max_output_tokens: int
    parallel: int = 1
    kv_cache_type: str = "f16"
    reasoning: str = "auto"         # on | off | auto
    reasoning_budget: Optional[int] = None
    num_gpus: Optional[int] = None  # required for MANUAL, ignored for AUTO

    @property
    def ctx_size(self) -> int:
        """Total llama.cpp --ctx-size: each of `parallel` slots gets in+out tokens."""
        return (self.max_input_tokens + self.max_output_tokens) * self.parallel

    @property
    def is_split_name(self) -> bool:
        """True if gguf_file looks like the first part of a split GGUF."""
        import re
        return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", self.gguf_file))


@dataclass
class Config:
    gpus_per_node: int
    models: list[ModelSpec] = field(default_factory=list)
    path: Optional[Path] = None

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

    hf_repo = _require(t, "hf_repo", where, str)
    gguf_file = _require(t, "gguf_file", where, str)

    alloc_mode = _require(t, "alloc_mode", where, str).upper()
    if alloc_mode not in ALLOC_MODES:
        raise ConfigError(f"{where}: alloc_mode must be one of {ALLOC_MODES}")

    max_in = _require(t, "max_input_tokens", where, int)
    max_out = _require(t, "max_output_tokens", where, int)
    if max_in <= 0 or max_out <= 0:
        raise ConfigError(f"{where}: max_input_tokens/max_output_tokens must be > 0")

    parallel = int(t.get("parallel", defaults.get("parallel", 1)))
    if parallel < 1:
        raise ConfigError(f"{where}: parallel must be >= 1")

    kv = str(t.get("kv_cache_type", defaults.get("kv_cache_type", "f16")))
    if kv not in KV_CACHE_TYPES:
        raise ConfigError(f"{where}: kv_cache_type must be one of {KV_CACHE_TYPES}")

    reasoning = str(t.get("reasoning", defaults.get("reasoning", "auto")))
    if reasoning not in REASONING_MODES:
        raise ConfigError(f"{where}: reasoning must be one of {REASONING_MODES}")

    reasoning_budget = t.get("reasoning_budget")
    if reasoning_budget is not None and (not isinstance(reasoning_budget, int)
                                         or isinstance(reasoning_budget, bool)):
        raise ConfigError(f"{where}: reasoning_budget must be an integer")

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
        kv_cache_type=kv, reasoning=reasoning, reasoning_budget=reasoning_budget,
        num_gpus=num_gpus,
    )


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

    return Config(gpus_per_node=gpus_per_node, models=models, path=path)


if __name__ == "__main__":  # quick manual check: python -m cerberus.config <file>
    cfg = load_config(sys.argv[1])
    print(f"gpus_per_node={cfg.gpus_per_node}, {len(cfg.models)} models")
    for m in cfg.models:
        print(f"  {m.label}: {m.alloc_mode} ctx={m.ctx_size} reasoning={m.reasoning} "
              f"num_gpus={m.num_gpus}")
