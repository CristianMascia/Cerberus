#!/usr/bin/env python3
"""
Cerberus (IBiSCo) — assegnazione delle GPU alle tre teste LLM.

Emette, per ciascun modello definito in models.conf, la GPU (indice logico
llama.cpp, o lista separata da virgola) su cui avviarlo. Il risultato è una riga
per modello, `nome<TAB>device`, letta da run_demo_ibisco.sh.

Due modalità (parametro --mode):

  manual : usa il campo 'cuda_device' di models.conf. Gli indici sono LOGICI,
           cioè relativi alle sole GPU assegnate dal job SLURM: se l'allocazione
           riceve le GPU fisiche 0,2,3, l'indice logico 0->0, 1->2, 2->3. La
           mappatura logico->fisico è gestita da SLURM (CUDA_VISIBLE_DEVICES) e
           da llama.cpp (--device CUDAn); qui ci si limita a validare che ogni
           indice sia minore del numero di GPU disponibili.

  auto   : ignora 'cuda_device' e usa 'est_mem_mb' (memoria stimata per modello).
           Interroga la memoria libera di ciascuna GPU dell'allocazione e assegna
           i modelli con un first-fit (bin-packing): più modelli piccoli possono
           condividere una GPU, un modello grande ne ottiene una capiente. Un
           modello che eccede la memoria di ogni singola GPU (es. oltre i 32 GB di
           una V100) viene DISTRIBUITO su più GPU: se ne sceglie un insieme la cui
           memoria libera combinata è sufficiente e si emette un '--tensor-split'
           proporzionale, così che ogni GPU ospiti una quota del modello coerente
           con la propria capienza. Se nemmeno l'insieme delle GPU basta,
           l'allocazione fallisce.

Formato di output (una riga per modello):

    nome<TAB>device<TAB>tensor_split

  - device       : indice/i logici della/e GPU (es. "2" oppure "1,3")
  - tensor_split : "-" per GPU singola; per lo split multi-GPU, le frazioni per
                   ciascun device nello stesso ordine (es. "0.55,0.45")
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Margine di sicurezza (MB) lasciato libero su ogni GPU in modalità auto,
# a copertura di overhead di contesto/KV-cache non incluso nella stima.
SAFETY_MARGIN_MB = 512


def eprint(*a):
    print("[allocate]", *a, file=sys.stderr)


def parse_models(path):
    """Ritorna la lista dei modelli con i campi rilevanti per l'allocazione."""
    models = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(";")]
        if len(fields) < 8:
            continue
        name, _repo, _glob, _port, _ctx, _maxtok, cuda_device, est_mem = fields[:8]
        models.append(
            {
                "name": name,
                "cuda_device": cuda_device,   # può essere "0" o "1,3"
                "est_mem_mb": int(est_mem),
            }
        )
    return models


def visible_physical_ids():
    """IDs fisici delle GPU visibili (da CUDA_VISIBLE_DEVICES), se interi."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return None
    toks = [t.strip() for t in vis.split(",") if t.strip() != ""]
    if all(t.isdigit() for t in toks):
        return [int(t) for t in toks]
    return None  # es. UUID: non gestiti, si ricade sull'ordine di nvidia-smi


def query_nvidia_smi():
    """Ritorna {indice_nvidia_smi: memoria_libera_MB} per le GPU riportate."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    smi = {}
    for line in out.stdout.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        smi[int(idx)] = int(free)
    return smi


def logical_free_mem():
    """Memoria libera (MB) per indice LOGICO llama.cpp delle GPU dell'allocazione."""
    smi = query_nvidia_smi()
    phys = visible_physical_ids()
    if phys is not None:
        # nvidia-smi mostra tutte le GPU fisiche: si filtra e riordina secondo
        # CUDA_VISIBLE_DEVICES (indice logico i -> GPU fisica phys[i]).
        if all(p in smi for p in phys):
            return [smi[p] for p in phys]
        # nvidia-smi mostra già solo le GPU allocate (isolamento cgroup): il suo
        # ordine coincide con quello logico.
        if len(smi) == len(phys):
            return [smi[i] for i in sorted(smi)]
    # Nessun CUDA_VISIBLE_DEVICES: ordine logico = ordine di nvidia-smi.
    return [smi[i] for i in sorted(smi)]


def num_logical_gpus():
    """Numero di GPU disponibili all'allocazione (senza richiedere memoria)."""
    phys = visible_physical_ids()
    if phys is not None:
        return len(phys)
    try:
        return len(query_nvidia_smi())
    except Exception:
        return None


def allocate_manual(models):
    n = num_logical_gpus()
    assignments = []
    for m in models:
        devices = [d.strip() for d in m["cuda_device"].split(",") if d.strip() != ""]
        if not devices:
            eprint(f"modello '{m['name']}': campo cuda_device vuoto (modalità manual).")
            sys.exit(2)
        for d in devices:
            if not d.isdigit():
                eprint(f"modello '{m['name']}': cuda_device '{d}' non è un intero.")
                sys.exit(2)
            if n is not None and int(d) >= n:
                eprint(
                    f"modello '{m['name']}': indice GPU {d} >= GPU disponibili ({n}). "
                    "Ridurre l'indice o richiedere più GPU al job."
                )
                sys.exit(2)
        # Split multi-GPU manuale: si lascia a llama.cpp la ripartizione di
        # default (uniforme) tra i device indicati.
        assignments.append((m["name"], ",".join(devices), "-"))
    return assignments


def allocate_auto(models):
    try:
        free = logical_free_mem()
    except FileNotFoundError:
        eprint("nvidia-smi non trovato: modalità auto non disponibile.")
        sys.exit(2)
    if not free:
        eprint("nessuna GPU rilevata.")
        sys.exit(2)

    remaining = list(free)
    assignments = []
    for m in models:
        need = m["est_mem_mb"]

        # 1) GPU singola: first-fit (preferito, evita l'overhead multi-GPU).
        placed = None
        for gi in range(len(remaining)):
            if remaining[gi] >= need + SAFETY_MARGIN_MB:
                remaining[gi] -= need
                placed = gi
                break
        if placed is not None:
            assignments.append((m["name"], str(placed), "-"))
            eprint(f"{m['name']} -> GPU {placed} (stima {need} MB, singola)")
            continue

        # 2) Split multi-GPU: si accumulano GPU (dalla più libera) finché la
        #    capacità utile combinata copre la memoria richiesta.
        order = sorted(range(len(remaining)), key=lambda g: remaining[g], reverse=True)
        chosen, capacity = [], 0
        for g in order:
            avail = remaining[g] - SAFETY_MARGIN_MB
            if avail <= 0:
                continue
            chosen.append(g)
            capacity += avail
            if capacity >= need:
                break

        if capacity < need:
            eprint(
                f"modello '{m['name']}' ({need} MB) non entra né in una singola GPU "
                f"né nell'insieme di quelle disponibili. "
                f"Memoria libera per GPU logica: {free} MB."
            )
            sys.exit(3)

        # Ripartizione proporzionale alla capacità utile di ciascuna GPU scelta.
        chosen.sort()  # ordine coerente tra --device e --tensor-split
        avails = [remaining[g] - SAFETY_MARGIN_MB for g in chosen]
        tot = sum(avails)
        fractions = [a / tot for a in avails]
        for g, frac in zip(chosen, fractions):
            remaining[g] -= need * frac

        device = ",".join(str(g) for g in chosen)
        tsplit = ",".join(f"{frac:.3f}" for frac in fractions)
        assignments.append((m["name"], device, tsplit))
        eprint(
            f"{m['name']} -> GPU {device} (stima {need} MB, split tensor {tsplit})"
        )
    return assignments


def main():
    ap = argparse.ArgumentParser(description="Assegna le GPU alle teste di Cerberus.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["manual", "auto"], required=True)
    args = ap.parse_args()

    models = parse_models(args.config)
    if not models:
        eprint(f"nessun modello valido in {args.config}")
        sys.exit(1)

    if args.mode == "manual":
        assignments = allocate_manual(models)
    else:
        assignments = allocate_auto(models)

    for name, device, tsplit in assignments:
        print(f"{name}\t{device}\t{tsplit}")


if __name__ == "__main__":
    main()
