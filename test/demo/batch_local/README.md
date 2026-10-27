# Cerberus — demo locale della batch API

Prova l'intero flusso di `cerberus batch` sul proprio PC, **senza SLURM**: un
`llama-server` con Qwen2.5-Coder-0.5B via Singularity e due shard eseguiti sullo
stesso server con `--endpoints`. Guida completa: [docs/batch.md](../../../docs/batch.md).

## Contenuto

| File | Ruolo |
|------|-------|
| `run_demo.sh` | Avvia il server, esegue validate → submit --dry-run → run (2 shard) → merge → controllo, arresta il server. |
| `models.conf` | Un modello (`qwen-coder`) e la sezione `[batch]`. |
| `requests.jsonl` | 4 richieste, una nel formato OpenAI Batch. |
| `../batch_check.py` | Verifica di `results.jsonl`. |
| `outputs/` | Cartelle dei batch (`batch_<timestamp>/`) ed `endpoints_<timestamp>.json`. |
| `logs/` | Log del `llama-server`. |

## Prerequisiti

1. Immagine `cerberus/containers/llamacpp_local.sif`
   (vedi [docs/installazione_container.md](../../../docs/installazione_container.md));
   altrimenti `CERBERUS_IMAGE=/percorso/immagine.sif`.
2. Pacchetto `cerberus` con le dipendenze (`pip install -e .` nella repo). Lo
   script usa `<repo>/.venv/bin/python` se esiste, altrimenti `python3`, o `PYTHON`.
3. **Rete**: `submit` legge l'header del GGUF da Hugging Face per stimare la VRAM.
   Se il modello non è in cache, lo script lo scarica con `cerberus download`.

## Esecuzione

```bash
./run_demo.sh
```

Esito atteso: `4 richieste x 2 ripetizioni = 8 risultati` da 2 shard, e alla fine
`[check] OK`. Il server è arrestato anche in caso di errore o Ctrl-C.

Variabili: `CERBERUS_IMAGE`, `HF_HOME`, `PYTHON`, `PORT` (default 8090), `NGL`,
`HEALTH_TIMEOUT`.
