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
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Blocco di ragionamento inline dei modelli "thinking" (es. Qwen3): <think>...</think>
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(content, reasoning):
    """Separa il ragionamento dalla risposta finale.

    Ritorna la coppia (thinking, answer):
    - se il server ha già isolato il ragionamento in 'reasoning_content', quello è
      il thinking e 'content' è la risposta;
    - altrimenti si estrae un eventuale blocco <think>...</think> dal content;
    - se il blocco <think> è aperto ma non chiuso (generazione troncata dentro il
      ragionamento), tutto ciò che segue <think> è thinking e la risposta è vuota.
    """
    content = content or ""
    if reasoning:
        return reasoning.strip(), content.strip()
    m = THINK_RE.search(content)
    if m:
        thinking = m.group(1).strip()
        answer = THINK_RE.sub("", content).strip()
        return thinking, answer
    if "<think>" in content and "</think>" not in content:
        return content.split("<think>", 1)[1].strip(), ""
    return "", content.strip()


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

    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    # I modelli "thinking" (es. Qwen3) espongono il ragionamento in un campo
    # separato; se non lo fanno, può essere inline come <think>...</think>.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    thinking, answer = split_thinking(content, reasoning)
    usage = body.get("usage", {})
    meta = {
        "latency_s": round(dt, 3),
        # 'length' = generazione troncata perché ha raggiunto max_tokens.
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return answer, thinking, meta


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
                answer, thinking, meta = query(
                    args.host, port, prompt, model["max_tokens"], args.timeout
                )
                # Thinking e risposta finale sono salvati in campi separati.
                entry.update(
                    {"thinking": thinking or None, "response": answer,
                     "error": None, **meta}
                )
                note = ""
                if thinking:
                    note += " [thinking separato]"
                if not answer and thinking:
                    note += " [risposta vuota: solo thinking]"
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
    lines = [f"# Cerberus — risposte ({ts})\n"]
    for run in results["runs"]:
        lines.append(f"## {run['model']} (porta {run['port']})\n")
        lines.append(f"**Prompt:** {run['prompt']}\n")
        if run["error"]:
            lines.append(f"**Errore:** `{run['error']}`\n")
        else:
            trunc = " — TRONCATA (max_tokens)" if run.get("finish_reason") == "length" else ""
            # Sezione thinking separata (solo se presente).
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
