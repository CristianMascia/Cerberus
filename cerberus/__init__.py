"""Cerberus — declarative multi-model llama.cpp serving on IBiSCo."""

from .config import Config, ConfigError, ModelSpec, load_config

__version__ = "0.1.0"

__all__ = ["Config", "ConfigError", "ModelSpec", "load_config", "__version__"]
