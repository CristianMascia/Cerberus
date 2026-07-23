#!/usr/bin/env python3
"""
Cerberus — client di interrogazione di un singolo modello LLM.

Invia tutti i prompt allo stesso modello servito da llama-server (endpoint
OpenAI-compatibile /v1/chat/completions) due volte: una in sequenza (una
richiesta alla volta) e una in parallelo (tutte le richieste inviate insieme
con un thread pool), misurando il tempo totale di ciascuna modalità.

Usa esclusivamente la libreria standard (urllib, concurrent.futures): non
richiede l'installazione di alcun pacchetto. Viene tipicamente invocato da
run_demo_local.sh, che prima avvia il server; può comunque essere lanciato a
mano se il server è già su.

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
from concurrent.futures import ThreadPoolExecutor
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


def run_one(host, port, max_tokens, timeout, prompt):
    """Esegue una singola query e ritorna sempre un entry, anche in caso di errore."""
    entry = {"prompt": prompt}
    try:
        answer, thinking, meta = query(host, port, prompt, max_tokens, timeout)
        entry.update(
            {"thinking": thinking or None, "response": answer, "error": None, **meta}
        )
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        entry.update({"thinking": None, "response": None, "error": str(exc)})
    return entry


def run_sequential(host, port, max_tokens, timeout, prompts):
    """Invia i prompt uno alla volta. Ritorna (entries, tempo_totale_s)."""
    print(f"=== sequenziale ({len(prompts)} richieste, una alla volta) ===")
    entries = []
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts, 1):
        entry = run_one(host, port, max_tokens, timeout, prompt)
        entries.append(entry)
        status = "ok" if entry["error"] is None else f"ERRORE: {entry['error']}"
        latency = entry.get("latency_s", "-")
        print(f"  [{i}/{len(prompts)}] {status}  ({latency}s)")
    dt = time.perf_counter() - t0
    print(f"  tempo totale: {dt:.3f}s\n")
    return entries, dt


def run_parallel(host, port, max_tokens, timeout, prompts):
    """Invia tutti i prompt in parallelo con un thread pool. Ritorna (entries, tempo_totale_s)."""
    print(f"=== parallelo ({len(prompts)} richieste, tutte insieme) ===")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        entries = list(
            pool.map(
                lambda prompt: run_one(host, port, max_tokens, timeout, prompt),
                prompts,
            )
        )
    dt = time.perf_counter() - t0
    for i, entry in enumerate(entries, 1):
        status = "ok" if entry["error"] is None else f"ERRORE: {entry['error']}"
        latency = entry.get("latency_s", "-")
        print(f"  [{i}/{len(prompts)}] {status}  ({latency}s)")
    print(f"  tempo totale: {dt:.3f}s\n")
    return entries, dt


def main():
    ap = argparse.ArgumentParser(description="Interroga un modello di Cerberus in sequenza e in parallelo.")
    ap.add_argument("--config", required=True, help="Percorso di models.conf")
    ap.add_argument("--prompts", required=True, help="Percorso di prompts.txt")
    ap.add_argument("--host", default="127.0.0.1", help="Host del server")
    ap.add_argument("--out", required=True, help="Directory di output")
    ap.add_argument("--timeout", type=float, default=300.0, help="Timeout richiesta (s)")
    args = ap.parse_args()

    models = parse_models(args.config)
    if not models:
        raise SystemExit(f"nessun modello valido in {args.config}")
    model = models[0]
    name, port, max_tokens = model["name"], model["port"], model["max_tokens"]

    prompts = parse_prompts(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[cerberus] modello '{name}' (porta {port}) — {len(prompts)} prompt\n")

    seq_entries, seq_time = run_sequential(args.host, port, max_tokens, args.timeout, prompts)
    par_entries, par_time = run_parallel(args.host, port, max_tokens, args.timeout, prompts)

    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print("=== confronto ===")
    print(f"  sequenziale: {seq_time:.3f}s")
    print(f"  parallelo:   {par_time:.3f}s")
    print(f"  speedup:     {speedup:.2f}x\n")

    results = {
        "timestamp": ts,
        "host": args.host,
        "model": name,
        "port": port,
        "timing": {
            "sequential_s": round(seq_time, 3),
            "parallel_s": round(par_time, 3),
            "speedup": round(speedup, 3),
        },
        "sequential_runs": seq_entries,
        "parallel_runs": par_entries,
    }

    # Salvataggio: JSON strutturato + Markdown leggibile.
    json_path = out_dir / f"responses_{ts}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def render_entries(lines, entries):
        for run in entries:
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

    md_path = out_dir / f"responses_{ts}.md"
    lines = [f"# Cerberus — risposte ({ts})\n"]
    lines.append(f"Modello: **{name}** (porta {port})\n")
    lines.append(
        f"Tempo sequenziale: **{seq_time:.3f}s** — "
        f"tempo parallelo: **{par_time:.3f}s** — speedup: **{speedup:.2f}x**\n"
    )
    lines.append("## Sequenziale\n")
    render_entries(lines, seq_entries)
    lines.append("## Parallelo\n")
    render_entries(lines, par_entries)
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[cerberus] risposte salvate in:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
