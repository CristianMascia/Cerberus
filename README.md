# Cerberus

**Serving dichiarativo di modelli LLM con `llama.cpp` sul cluster HPC IBiSCo**
(SLURM, Singularity, GPU NVIDIA V100-SXM2 da 32 GB, filesystem Lustre).

Scrivi **un solo file** — `models.conf` (TOML) — in cui dichiari *quali* modelli
vuoi servire e *quanto* occupano. Cerberus fa il resto: scarica i GGUF, decide
quante GPU serve a ogni modello, li posiziona su nodi e GPU dell'allocazione,
avvia i server `llama-server` e genera una **mappa degli endpoint** che un client
Python usa per instradare le richieste **per etichetta (label)** tramite l'API
compatibile con OpenAI.

Cerberus è pensato per essere **semplice da usare**: `clona → configura → lancia`.

## Avvio rapido

```bash
# 1. ambiente (una volta)
conda create -n cerberus python=3.11 -y && conda activate cerberus
pip install -e .                     # installa il comando `cerberus` e le dipendenze
export HF_TOKEN=hf_...

# 2. configura e verifica
cerberus validate                    # valida models.conf, stima GPU/nodi necessari
cerberus download                    # scarica i GGUF nella cache HF condivisa

# 3. alloca le risorse e avvia
salloc --partition=gpus --nodes=<N> --ntasks-per-node=1 --gpus-per-node=3 ...
cerberus up                          # posiziona, avvia i server, scrive endpoints.json
```

```python
# 4. usa i modelli dal tuo codice
from cerberus import CerberusClient
c = CerberusClient()
print(c.chat("qwen-big", [{"role": "user", "content": "Ciao"}]).content)
```

## Architettura in breve

```
models.conf ──► cerberus validate / download / up ──► endpoints.json ──► CerberusClient
  (tu)            (config → VRAM → placement →            (label →          (il tuo
                   SLURM/Singularity → server)             host:port)        codice)
```

Componenti principali (tutti dentro il package `cerberus/`):

- **config** — parsing e validazione di `models.conf`.
- **estimator** — stima VRAM leggendo gli header GGUF (senza scaricare).
- **vram / placement** — decide quante GPU servono (AUTO) e impacchetta i modelli
  su nodi e GPU (condivisione dei piccoli, tensor-split intra-nodo dei grandi).
- **download** — scarica i GGUF esatti (via `huggingface_hub`).
- **deploy + node_launch.sh** — avvia i `llama-server` nella sandbox CUDA tramite
  `srun`, uno per nodo, e genera la mappa degli endpoint.
- **client** — client OpenAI-compatibile che instrada per label e separa il
  *reasoning* dalla risposta.

## Struttura della repository

| Percorso | Contenuto |
|----------|-----------|
| `cerberus/` | il tool e **tutti** i suoi componenti in un unico posto |
| `cerberus/estimator.py` | stimatore VRAM/GGUF (header GGUF via HTTP, no download) |
| `cerberus/containers/` | ricette Singularity per costruire immagine/sandbox |
| `cerberus/node_launch.sh` | launcher per-nodo (eseguito via `srun`) |
| `test/demo/` | demo eseguibili: `ibisco` (1 nodo), `ibisco_multinode` (2 nodi), `local` (immagine locale) |
| `test/unit/` | test unitari dei componenti in isolamento |
| `docs/` | guide dettagliate (vedi sotto) |
| `models.conf.example` | esempio annotato di configurazione |
| `pyproject.toml`, `environment.yml` | packaging e ambiente conda |

## Documentazione (in `docs/`)

Le procedure dettagliate NON sono in questo README ma nelle guide:

- **[docs/panoramica.md](docs/panoramica.md)** — cosa offre Cerberus e come funziona,
  nel dettaglio (schema `models.conf`, AUTO/MANUAL, placement, argomenti dei server,
  variabili d'ambiente).
- **[docs/progetto.md](docs/progetto.md)** — come realizzare un tuo progetto:
  scrivere `models.conf`, allocare le risorse, avviare i modelli e **integrare il
  client** nel tuo codice.
- **[docs/demo.md](docs/demo.md)** — eseguire le demo passo-passo e diagnosi degli errori.
- **[docs/installazione_container.md](docs/installazione_container.md)** — build
  dell'immagine `.sif` (locale) e della sandbox CUDA (IBiSCo).
- **[docs/download_modelli.md](docs/download_modelli.md)** — download dei modelli da Hugging Face.
- **[docs/uso_llamacpp.md](docs/uso_llamacpp.md)** — uso manuale di `llama.cpp` in Singularity.

## Test

```bash
python -m unittest discover -s test/unit          # test unitari (nessuna dipendenza pesante)
```
Le demo in `test/demo/` sono test d'integrazione eseguibili su IBiSCo — vedi
[docs/demo.md](docs/demo.md).

## Requisiti

Python ≥ 3.11; sul cluster: sandbox CUDA e binari di `llama.cpp` compilati per
sm_70 (vedi [docs/installazione_container.md](docs/installazione_container.md)).
