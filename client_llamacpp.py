"""
Cerberus — client for the served llama.cpp models (OpenAI-compatible API).

Single entry point to every model deployed by `cerberus up`: you pick a model by
its **label** and the client resolves which `host:port` to talk to from the
generated `endpoints.json`.

Off-cluster safe: importing this module and constructing `CerberusClient()` never
touch the filesystem or network. The endpoint map is loaded lazily on first use;
off-cluster (no map) `is_available()` returns False and calls raise
`CerberusUnavailable`.

Reasoning: models served with `--reasoning-format deepseek` expose their thinking
in `message.reasoning_content`; the client returns a clean `content` and the trace
separately in `reasoning`. A `<think>...</think>` fallback covers other cases.

Example
-------
    from client_llamacpp import CerberusClient
    c = CerberusClient()
    r = c.chat("qwen-big", [{"role": "user", "content": "Ciao"}], reasoning=False)
    print(r.content)      # clean answer
    print(r.reasoning)    # thinking trace or None
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class CerberusUnavailable(RuntimeError):
    """Raised when the endpoint map is not available (e.g. running off-cluster)."""


@dataclass
class Response:
    content: str                 # clean final answer (never contains <think>)
    reasoning: Optional[str]     # reasoning trace, or None
    finish_reason: Optional[str]
    raw: Any                     # full SDK response object


def _split_thinking(content: str, reasoning: Optional[str]) -> tuple[Optional[str], str]:
    content = content or ""
    if reasoning:
        return reasoning.strip() or None, content.strip()
    m = _THINK_RE.search(content)
    if m:
        return m.group(1).strip() or None, _THINK_RE.sub("", content).strip()
    if "<think>" in content and "</think>" not in content:
        return content.split("<think>", 1)[1].strip() or None, ""
    return None, content.strip()


class CerberusClient:
    def __init__(self, endpoints_path: Optional[str] = None,
                 project_dir: Optional[str] = None):
        self._explicit = endpoints_path
        self._project_dir = project_dir
        self._map: Optional[dict] = None
        self._sdk_clients: dict[str, Any] = {}

    # --- endpoint map (lazy) ------------------------------------------------ #
    def _map_path(self) -> Path:
        if self._explicit:
            return Path(self._explicit)
        env = os.environ.get("CERBERUS_ENDPOINTS")
        if env:
            return Path(env)
        base = Path(self._project_dir) if self._project_dir else Path.cwd()
        return base / "endpoints.json"

    def _load(self) -> dict:
        if self._map is None:
            path = self._map_path()
            if not path.is_file():
                raise CerberusUnavailable(
                    f"endpoint map not found at {path} — run 'cerberus up', set "
                    "CERBERUS_ENDPOINTS, or pass endpoints_path/project_dir"
                )
            self._map = json.loads(path.read_text())["models"]
        return self._map

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except CerberusUnavailable:
            return False

    def list_models(self) -> list[str]:
        return list(self._load().keys())

    def endpoint(self, label: str) -> dict:
        models = self._load()
        if label not in models:
            raise KeyError(f"unknown model label '{label}'; have {list(models)}")
        return models[label]

    # --- OpenAI SDK per model ---------------------------------------------- #
    def _sdk(self, label: str):
        if label not in self._sdk_clients:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("the 'openai' package is required (pip install openai)") from exc
            self._sdk_clients[label] = OpenAI(base_url=self.endpoint(label)["base_url"],
                                              api_key="cerberus-no-auth")
        return self._sdk_clients[label]

    # --- inference ---------------------------------------------------------- #
    def chat(self, label: str, messages: list[dict], *, reasoning: Optional[bool] = None,
             max_tokens: Optional[int] = None, temperature: float = 0.7,
             stream: bool = False, **extra) -> Response:
        """Send a chat request to the model `label`.

        reasoning: None = leave the server default; True/False = best-effort ask the
        model to think / not think (via chat_template_kwargs).
        """
        client = self._sdk(label)
        extra_body = dict(extra.pop("extra_body", {}) or {})
        if reasoning is not None:
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(reasoning)

        resp = client.chat.completions.create(
            model=label, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=stream,
            extra_body=extra_body or None, **extra,
        )
        if stream:  # advanced use: hand back the raw stream
            return resp  # type: ignore[return-value]

        choice = resp.choices[0]
        msg = choice.message
        raw_content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None and getattr(msg, "model_extra", None):
            reasoning_content = msg.model_extra.get("reasoning_content")
        trace, answer = _split_thinking(raw_content, reasoning_content)
        return Response(content=answer, reasoning=trace,
                        finish_reason=choice.finish_reason, raw=resp)


if __name__ == "__main__":  # tiny smoke test / demo
    import sys
    c = CerberusClient()
    if not c.is_available():
        sys.exit("no endpoint map (run this on IBiSCo after 'cerberus up')")
    print("models:", c.list_models())
    label = sys.argv[1] if len(sys.argv) > 1 else c.list_models()[0]
    r = c.chat(label, [{"role": "user", "content": "Ciao, chi sei?"}])
    if r.reasoning:
        print("[reasoning]", r.reasoning[:200], "...")
    print("[answer]", r.content)
