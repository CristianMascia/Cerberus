#!/usr/bin/env python3
"""
bench.py — benchmark di un modello servito da Cerberus (llama.cpp o vLLM).

Stesso carico ovunque (IBiSCo, HPC UniNA, llama.cpp, vLLM), così i numeri sono
confrontabili: prompt sintetici con seme fisso, temperatura 0, uscita di
lunghezza fissa (ignore_eos), thinking spento, richieste in streaming per
misurare il tempo al primo token. Per ogni livello di concorrenza scrive una
riga JSON nel file dei risultati.

Uso (dove c'è endpoints.json, con l'env cerberus attivo):
  python scripts/bench.py run --model qwen3-8b --tag unina-vllm
  python scripts/bench.py run --model qwen3-8b --tag ibisco-llamacpp --concurrency 1,4,16,32
  python scripts/bench.py compare bench_results.jsonl

Metriche: throughput in uscita (token/s di tutto il server), richieste/s,
TTFT (tempo al primo token), TPOT (tempo per token dopo il primo), latenza
totale; p50 e p95.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

WORDS = ("analisi modello rete dati calcolo nodo memoria tempo valore funzione "
         "sistema risultato processo codice errore misura carico serie studio "
         "metodo campione parametro struttura livello segnale energia flusso "
         "ordine stato evento regola grafo vettore matrice insieme limite").split()


def make_prompts(n: int, in_tokens: int, seed: int) -> list[str]:
    """`n` prompt diversi di circa `in_tokens` token (parole italiane: ~1.8 token l'una).

    Il conteggio vero, dal server, finisce in mean_in_tokens.
    """
    rng = random.Random(seed)
    words = max(1, int(in_tokens * 0.55))
    return [f"Documento {i}.\n" + " ".join(rng.choice(WORDS) for _ in range(words))
            + "\n\nRiassumi il documento in italiano, in modo dettagliato."
            for i in range(n)]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


async def one(client, model: str, prompt: str, max_tokens: int, extra: dict) -> dict:
    t0 = time.perf_counter()
    ttft, chunks, usage = None, 0, None
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
        stream_options={"include_usage": True}, extra_body=extra)
    async for ch in stream:
        if getattr(ch, "usage", None):
            usage = ch.usage
        if not ch.choices:
            continue
        d = ch.choices[0].delta
        extra_d = getattr(d, "model_extra", None) or {}
        text = (d.content or extra_d.get("reasoning_content") or extra_d.get("reasoning"))
        if text:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    e2e = time.perf_counter() - t0
    out = usage.completion_tokens if usage and usage.completion_tokens else chunks
    return {"ttft": ttft if ttft is not None else e2e, "e2e": e2e, "out": out,
            "in": usage.prompt_tokens if usage else None,
            "tpot": (e2e - ttft) / (out - 1) if ttft is not None and out > 1 else None}


async def level(client, model, prompts, conc, max_tokens, extra) -> dict:
    sem = asyncio.Semaphore(conc)

    async def guarded(p):
        async with sem:
            try:
                return await one(client, model, p, max_tokens, extra)
            except Exception as exc:  # noqa: BLE001 - un errore conta, non ferma il livello
                return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(p) for p in prompts))
    wall = time.perf_counter() - t0
    ok = [r for r in res if "error" not in r]
    errs = [r["error"] for r in res if "error" in r]
    out_tok = sum(r["out"] for r in ok)
    ms = lambda xs: [1000 * x for x in xs]  # noqa: E731
    ttft = ms([r["ttft"] for r in ok])
    tpot = ms([r["tpot"] for r in ok if r["tpot"] is not None])
    e2e = [r["e2e"] for r in ok]
    ins = [r["in"] for r in ok if r["in"]]
    return {
        "concurrency": conc, "requests": len(prompts), "errors": len(errs),
        "first_error": errs[0] if errs else None, "wall_s": round(wall, 2),
        "output_tok_s": round(out_tok / wall, 1) if wall else None,
        "req_s": round(len(ok) / wall, 3) if wall else None,
        "mean_in_tokens": round(statistics.mean(ins)) if ins else None,
        "mean_out_tokens": round(out_tok / len(ok)) if ok else None,
        "ttft_ms_p50": pct(ttft, .5), "ttft_ms_p95": pct(ttft, .95),
        "tpot_ms_p50": pct(tpot, .5), "tpot_ms_p95": pct(tpot, .95),
        "e2e_s_p50": pct(e2e, .5), "e2e_s_p95": pct(e2e, .95),
    }


def load_endpoint(path: str | None, label: str) -> dict:
    p = Path(path or os.environ.get("CERBERUS_ENDPOINTS") or "endpoints.json")
    if not p.is_file():
        sys.exit(f"endpoints.json non trovato in {p} (lancia prima cerberus up, o --endpoints)")
    models = json.loads(p.read_text())["models"]
    if label not in models:
        sys.exit(f"modello '{label}' non in {p}; ci sono {list(models)}")
    return models[label]


def fmt(v, nd=0):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def print_rows(rows: list[dict]) -> None:
    cols = [("tag", "tag", 0), ("backend", "backend", 0), ("conc", "concurrency", 0),
            ("tok/s", "output_tok_s", 0), ("req/s", "req_s", 2),
            ("TTFT p50 ms", "ttft_ms_p50", 0), ("TTFT p95 ms", "ttft_ms_p95", 0),
            ("TPOT p50 ms", "tpot_ms_p50", 1), ("e2e p50 s", "e2e_s_p50", 1),
            ("err", "errors", 0)]
    table = [[h for h, _, _ in cols]] + [[fmt(r.get(k), nd) for _, k, nd in cols] for r in rows]
    w = [max(len(row[i]) for row in table) for i in range(len(cols))]
    for row in table:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(row)))


async def run(args) -> None:
    from openai import AsyncOpenAI

    ep = load_endpoint(args.endpoints, args.model)
    backend = ep.get("backend", "llamacpp")  # endpoints.json vecchi: solo llama.cpp
    client = AsyncOpenAI(base_url=ep["base_url"], api_key="cerberus-no-auth",
                         timeout=args.timeout, max_retries=0)
    extra = {"chat_template_kwargs": {"enable_thinking": args.thinking}}
    if not args.allow_eos:
        extra["ignore_eos"] = True  # stessa lunghezza d'uscita su entrambi i backend
    concs = [int(c) for c in args.concurrency.split(",")]

    print(f"[bench] {args.model} ({backend}) su {ep['base_url']}; "
          f"~{args.input_tokens} token in, {args.output_tokens} out; concorrenze {concs}")
    warm = make_prompts(2, args.input_tokens, args.seed + 999)
    await asyncio.gather(*(one(client, args.model, p, 16, extra) for p in warm))

    rows = []
    out = Path(args.out)
    for c in concs:
        n = args.requests or max(16, 4 * c)
        prompts = make_prompts(n, args.input_tokens, args.seed + c)
        r = await level(client, args.model, prompts, c, args.output_tokens, extra)
        r.update({
            "tag": args.tag, "model": args.model, "backend": backend,
            "server_host": ep.get("host"), "client_host": socket.gethostname(),
            "gpu": args.gpu, "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens, "ignore_eos": not args.allow_eos,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        rows.append(r)
        with out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print_rows([r])
        if r["errors"]:
            print(f"  primo errore: {r['first_error']}")
    print(f"[bench] risultati aggiunti a {out}")


def compare(args) -> None:
    rows = []
    for f in args.files:
        rows += [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (r.get("input_tokens"), r.get("output_tokens"),
                             r["concurrency"], r.get("tag") or ""))
    print_rows(rows)
    base = [r for r in rows if r.get("tag") == args.baseline] if args.baseline else []
    if base:
        print(f"\nspeedup di throughput rispetto a '{args.baseline}' (stessa concorrenza e carico):")
        ref = {(r["input_tokens"], r["output_tokens"], r["concurrency"]): r for r in base}
        for r in rows:
            b = ref.get((r["input_tokens"], r["output_tokens"], r["concurrency"]))
            if b and r.get("tag") != args.baseline and b.get("output_tok_s"):
                print(f"  {r['tag']:>20}  conc {r['concurrency']:>4}  "
                      f"x{r['output_tok_s'] / b['output_tok_s']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="misura un modello di endpoints.json")
    r.add_argument("--model", required=True, help="label del modello in endpoints.json")
    r.add_argument("--tag", required=True, help="nome della prova, es. unina-vllm")
    r.add_argument("--endpoints", help="default: $CERBERUS_ENDPOINTS o ./endpoints.json")
    r.add_argument("--concurrency", default="1,4,16,32,64")
    r.add_argument("--requests", type=int, default=0,
                   help="richieste per livello (default max(16, 4 x concorrenza))")
    r.add_argument("--input-tokens", type=int, default=1024)
    r.add_argument("--output-tokens", type=int, default=256)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--gpu", default=os.environ.get("BENCH_GPU", ""), help="es. V100, H100")
    r.add_argument("--thinking", action="store_true", help="lascia il thinking acceso")
    r.add_argument("--allow-eos", action="store_true", help="non forzare la lunghezza d'uscita")
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--out", default="bench_results.jsonl")
    c = sub.add_parser("compare", help="tabella dai file di risultati")
    c.add_argument("files", nargs="+")
    c.add_argument("--baseline", help="tag di riferimento per lo speedup, es. ibisco-llamacpp")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        compare(args)


if __name__ == "__main__":
    main()
