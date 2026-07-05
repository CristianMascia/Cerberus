"""Cerberus — serving dichiarativo multi-modello di llama.cpp su IBiSCo."""

from .config import Config, ConfigError, ModelSpec, load_config
from .client import CerberusClient, CerberusUnavailable, Response

__version__ = "0.2.0"

__all__ = [
    "Config", "ConfigError", "ModelSpec", "load_config",
    "CerberusClient", "CerberusUnavailable", "Response",
    "__version__",
]
