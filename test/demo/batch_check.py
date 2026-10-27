#!/usr/bin/env python3
"""
Cerberus — controllo dei risultati di un batch (usato dalle demo batch_*).

Verifica che <batch-dir>/results.jsonl (scritto da `cerberus batch merge`):
  * abbia una riga per ogni (custom_id, ripetizione), nell'ordine dell'input;
  * non contenga errori;
  * riporti per ogni riga il modello richiesto e una risposta non vuota;
  * contenga risultati di tutti gli shard.

Uso:  python batch_check.py --batch-dir DIR [--expect N]
Esce con codice 1 al primo controllo fallito.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--expect", type=int, help="numero di risultati atteso "
                    "(default: richieste x ripetizioni dal manifest)")
    args = ap.parse_args(argv)

    bd = Path(args.batch_dir)
    manifest = json.loads((bd / "manifest.json").read_text())
    requests = _jsonl(bd / "requests.jsonl")
    results_path = bd / "results.jsonl"
    if not results_path.is_file():
        print(f"[check] FAIL: {results_path} non esiste (eseguire 'cerberus batch merge')")
        return 1
    results = _jsonl(results_path)

    failures = []
    expect = args.expect if args.expect is not None else manifest["n_units"]
    if len(results) != expect:
        failures.append(f"{len(results)} risultati, attesi {expect}")

    order = [(r["custom_id"], k) for r in requests for k in range(manifest["repeats"])]
    got = [(r["custom_id"], r["repeat"]) for r in results]
    if got != order:
        failures.append("ordine/insieme dei risultati diverso dall'input")

    model_of = {r["custom_id"]: r["model"] for r in requests}
    for r in results:
        key = f"{r['custom_id']}#{r['repeat']}"
        if r.get("error"):
            failures.append(f"{key}: errore {r['error']}")
            continue
        if r.get("model") != model_of.get(r["custom_id"]):
            failures.append(f"{key}: modello {r.get('model')}, atteso "
                            f"{model_of.get(r['custom_id'])}")
        content = ((r.get("response") or {}).get("content") or "").strip()
        if not content:
            failures.append(f"{key}: risposta vuota")

    shards = {r.get("shard") for r in results}
    if shards != set(range(manifest["shards"])):
        failures.append(f"shard presenti {sorted(shards)}, attesi "
                        f"{list(range(manifest['shards']))}")

    if failures:
        for f in failures:
            print(f"[check] FAIL: {f}")
        return 1

    print(f"[check] OK — {len(results)} risultati da {len(shards)} shard, "
          f"{len(requests)} richieste x {manifest['repeats']} ripetizioni")
    for r in results[:3]:
        text = r["response"]["content"].replace("\n", " ")
        print(f"  {r['custom_id']}#{r['repeat']} (shard {r['shard']}, "
              f"{r['latency_s']}s): {text[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
