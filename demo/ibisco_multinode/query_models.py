#!/usr/bin/env python3
"""
Cerberus multinode — client di interrogazione dei modelli su più nodi.

Come il client single-node, ma ogni modello vive su un nodo diverso: gli endpoint
(nome, host, porta, max_tokens) sono letti da un file TSV generato dal master,
invece di assumere tutti i server su 127.0.0.1. Interroga i modelli in sequenza e
salva le risposte (thinking e risposta finale separati) in JSON e Markdown.

Usa solo la libreria standard (urllib). Tipicamente invocato da
run_demo_multinode.sh dopo l'avvio dei server sui nodi.

Esempio:
    python3 query_models.py --endpoints .endpoints.tsv --prompts prompts.txt --out outputs
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Blocco di ragionamento inline dei modelli "thinking" (es. Qwen3): <think>...</think>
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(content, reasoning):
    """Separa il ragionamento dalla risposta finale; ritorna (thinking, answer)."""
    content = content or ""
    if reasoning:
        return reasoning.strip(), content.strip()
    m = THINK_RE.search(content)
    if m:
        return m.group(1).strip(), THINK_RE.sub("", content).strip()
    if "<think>" in content and "</think>" not in content:
        return content.split("<think>", 1)[1].strip(), ""
    return "", content.strip()


def parse_endpoints(path):
    """Legge il TSV degli endpoint: nome<TAB>host<TAB>porta<TAB>max_tokens."""
    eps = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        name, host, port, max_tokens = fields[:4]
        eps.append(
            {"name": name, "host": host, "port": int(port),
             "max_tokens": int(max_tokens)}
        )
    return eps


def parse_prompts(path):
    prompts = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def query(host, port, prompt, max_tokens, timeout):
    """Invia un prompt all'endpoint OpenAI-compatibile; ritorna (answer, thinking, meta)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "local",  # ignorato: il server espone l'unico modello caricato
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

    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    thinking, answer = split_thinking(content, reasoning)
    usage = body.get("usage", {})
    meta = {
        "latency_s": round(dt, 3),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return answer, thinking, meta


def main():
    ap = argparse.ArgumentParser(description="Interroga i modelli multinode di Cerberus.")
    ap.add_argument("--endpoints", required=True, help="TSV: nome host porta max_tokens")
    ap.add_argument("--prompts", required=True, help="Percorso di prompts.txt")
    ap.add_argument("--out", required=True, help="Directory di output")
    ap.add_argument("--timeout", type=float, default=300.0, help="Timeout richiesta (s)")
    args = ap.parse_args()

    endpoints = parse_endpoints(args.endpoints)
    prompts = parse_prompts(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {"timestamp": ts, "runs": []}

    print(f"[cerberus] {len(endpoints)} modelli x {len(prompts)} prompt\n")

    # In sequenza: per ciascun modello (sul suo nodo), tutti i prompt.
    for ep in endpoints:
        name, host, port = ep["name"], ep["host"], ep["port"]
        print(f"=== {name}  ({host}:{port}) ===")
        for i, prompt in enumerate(prompts, 1):
            entry = {"model": name, "host": host, "port": port, "prompt": prompt}
            try:
                answer, thinking, meta = query(
                    host, port, prompt, ep["max_tokens"], args.timeout
                )
                entry.update(
                    {"thinking": thinking or None, "response": answer,
                     "error": None, **meta}
                )
                note = ""
                if thinking:
                    note += " [thinking separato]"
                if meta.get("finish_reason") == "length":
                    note += " [troncato: raggiunto max_tokens]"
                print(f"  [{i}/{len(prompts)}] ok  ({meta['latency_s']}s){note}")
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                entry.update({"thinking": None, "response": None, "error": str(exc)})
                print(f"  [{i}/{len(prompts)}] ERRORE: {exc}")
            results["runs"].append(entry)
        print()

    # Salvataggio: JSON strutturato + Markdown leggibile.
    json_path = out_dir / f"responses_{ts}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_path = out_dir / f"responses_{ts}.md"
    lines = [f"# Cerberus multinode — risposte ({ts})\n"]
    for run in results["runs"]:
        lines.append(f"## {run['model']} ({run['host']}:{run['port']})\n")
        lines.append(f"**Prompt:** {run['prompt']}\n")
        if run["error"]:
            lines.append(f"**Errore:** `{run['error']}`\n")
        else:
            trunc = " — TRONCATA (max_tokens)" if run.get("finish_reason") == "length" else ""
            if run.get("thinking"):
                lines.append("**Thinking:**\n")
                lines.append(f"```\n{run['thinking']}\n```\n")
            lines.append(
                f"**Risposta** ({run.get('latency_s')}s, "
                f"{run.get('completion_tokens')} token{trunc}):\n"
            )
            answer = run.get("response") or ""
            lines.append(f"{answer}\n" if answer.strip() else "_(risposta vuota)_\n")
        lines.append("---\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[cerberus] risposte salvate in:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
