#!/usr/bin/env python3
"""
Cerberus — client di interrogazione delle tre teste LLM.

Invia, in sequenza, ogni prompt a ciascun modello servito da llama-server
(endpoint OpenAI-compatibile /v1/chat/completions) e salva le risposte.

Usa esclusivamente la libreria standard (urllib): non richiede l'installazione
di alcun pacchetto. Viene tipicamente invocato da run_demo_local.sh, che prima
avvia i tre server; può comunque essere lanciato a mano se i server sono già su.

Esempio:
    python3 query_models.py \
        --config models.conf \
        --prompts prompts.txt \
        --host 127.0.0.1 \
        --out outputs
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_models(path):
    """Legge models.conf e ritorna la lista dei modelli (nome, porta, max_tokens)."""
    models = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(";")]
        if len(fields) < 6:
            continue
        name, _repo, _glob, port, _ctx, max_tokens = fields[:6]
        models.append(
            {"name": name, "port": int(port), "max_tokens": int(max_tokens)}
        )
    return models


def parse_prompts(path):
    """Legge prompts.txt e ritorna la lista dei prompt."""
    prompts = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def query(host, port, prompt, max_tokens, timeout):
    """Invia un prompt all'endpoint OpenAI-compatibile e ritorna (testo, meta)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        # 'model' è ignorato: il server espone l'unico modello caricato.
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0

    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    meta = {
        "latency_s": round(dt, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return content, meta


def main():
    ap = argparse.ArgumentParser(description="Interroga le tre teste di Cerberus.")
    ap.add_argument("--config", required=True, help="Percorso di models.conf")
    ap.add_argument("--prompts", required=True, help="Percorso di prompts.txt")
    ap.add_argument("--host", default="127.0.0.1", help="Host dei server")
    ap.add_argument("--out", required=True, help="Directory di output")
    ap.add_argument("--timeout", type=float, default=300.0, help="Timeout richiesta (s)")
    args = ap.parse_args()

    models = parse_models(args.config)
    prompts = parse_prompts(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {"timestamp": ts, "host": args.host, "runs": []}

    print(f"[cerberus] {len(models)} modelli x {len(prompts)} prompt\n")

    # In sequenza: per ciascun modello, tutti i prompt.
    for model in models:
        name, port = model["name"], model["port"]
        print(f"=== {name}  (porta {port}) ===")
        for i, prompt in enumerate(prompts, 1):
            entry = {
                "model": name,
                "port": port,
                "prompt": prompt,
            }
            try:
                content, meta = query(
                    args.host, port, prompt, model["max_tokens"], args.timeout
                )
                entry.update({"response": content, "error": None, **meta})
                print(f"  [{i}/{len(prompts)}] ok  ({meta['latency_s']}s)")
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                entry.update({"response": None, "error": str(exc)})
                print(f"  [{i}/{len(prompts)}] ERRORE: {exc}")
            results["runs"].append(entry)
        print()

    # Salvataggio: JSON strutturato + Markdown leggibile.
    json_path = out_dir / f"responses_{ts}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_path = out_dir / f"responses_{ts}.md"
    lines = [f"# Cerberus — risposte ({ts})\n"]
    for run in results["runs"]:
        lines.append(f"## {run['model']} (porta {run['port']})\n")
        lines.append(f"**Prompt:** {run['prompt']}\n")
        if run["error"]:
            lines.append(f"**Errore:** `{run['error']}`\n")
        else:
            lines.append(
                f"**Risposta** ({run.get('latency_s')}s, "
                f"{run.get('completion_tokens')} token):\n"
            )
            lines.append(f"{run['response']}\n")
        lines.append("---\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[cerberus] risposte salvate in:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
