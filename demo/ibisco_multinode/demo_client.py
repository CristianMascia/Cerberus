#!/usr/bin/env python3
"""
Cerberus demo client — queries every deployed model via the Cerberus client.

Reads the endpoint map that `cerberus up` wrote in this directory (endpoints.json),
sends each prompt in prompts.txt to every model in sequence, and saves the answers
(thinking separated from the final answer) to outputs/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
# make the repo-root client_llamacpp importable without installing
sys.path.insert(0, str(HERE.parents[1]))
from client_llamacpp import CerberusClient  # noqa: E402


def read_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main():
    c = CerberusClient(project_dir=str(HERE))
    if not c.is_available():
        sys.exit("no endpoints.json here — run 'cerberus up' first (in this dir).")

    prompts = read_prompts(HERE / "prompts.txt")
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = []

    models = c.list_models()
    print(f"[demo] {len(models)} model(s) x {len(prompts)} prompt(s)\n")
    for label in models:
        print(f"=== {label} ({c.endpoint(label)['base_url']}) ===")
        for i, prompt in enumerate(prompts, 1):
            entry = {"model": label, "prompt": prompt}
            try:
                t0 = time.perf_counter()
                r = c.chat(label, [{"role": "user", "content": prompt}], max_tokens=512)
                dt = round(time.perf_counter() - t0, 2)
                entry.update({"thinking": r.reasoning, "response": r.content,
                              "finish_reason": r.finish_reason, "latency_s": dt, "error": None})
                note = " [thinking]" if r.reasoning else ""
                print(f"  [{i}/{len(prompts)}] ok ({dt}s){note}")
            except Exception as exc:  # noqa: BLE001
                entry.update({"thinking": None, "response": None, "error": str(exc)})
                print(f"  [{i}/{len(prompts)}] ERROR: {exc}")
            runs.append(entry)
        print()

    (out_dir / f"responses_{ts}.json").write_text(
        json.dumps({"timestamp": ts, "runs": runs}, ensure_ascii=False, indent=2))

    md = [f"# Cerberus demo — risposte ({ts})\n"]
    for r in runs:
        md.append(f"## {r['model']}\n")
        md.append(f"**Prompt:** {r['prompt']}\n")
        if r["error"]:
            md.append(f"**Errore:** `{r['error']}`\n")
        else:
            if r.get("thinking"):
                md.append("**Thinking:**\n")
                md.append(f"```\n{r['thinking']}\n```\n")
            md.append(f"**Risposta** ({r.get('latency_s')}s):\n")
            md.append((r["response"] or "_(vuota)_") + "\n")
        md.append("---\n")
    (out_dir / f"responses_{ts}.md").write_text("\n".join(md))
    print(f"[demo] saved outputs/responses_{ts}.json / .md")


if __name__ == "__main__":
    main()
