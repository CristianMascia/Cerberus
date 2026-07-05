#!/usr/bin/env python3
"""
Client condiviso delle demo — interroga ogni modello servito da Cerberus.

Legge la mappa `endpoints.json` scritta da `cerberus up` nella cartella CORRENTE
(quella della demo da cui lo lanci), invia ogni prompt di `prompts.txt` a tutti i
modelli in sequenza e salva le risposte (thinking separato dalla risposta finale)
in `outputs/`.

Uso (dalla cartella di una demo):  python ../demo_client.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from cerberus import CerberusClient   # installato con `pip install -e .`

CWD = Path.cwd()


def read_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main():
    c = CerberusClient(project_dir=str(CWD))
    if not c.is_available():
        sys.exit("nessun endpoints.json qui — esegui prima 'cerberus up' in questa cartella.")

    prompts = read_prompts(CWD / "prompts.txt")
    out_dir = CWD / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = []

    models = c.list_models()
    print(f"[demo] {len(models)} modello/i x {len(prompts)} prompt\n")
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
                print(f"  [{i}/{len(prompts)}] ERRORE: {exc}")
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
    print(f"[demo] salvato outputs/responses_{ts}.json / .md")


if __name__ == "__main__":
    main()
