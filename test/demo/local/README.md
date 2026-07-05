# Cerberus — demo locale (tre teste LLM)

Esempio di progetto che avvia **tre modelli LLM** con `llama.cpp` su Singularity,
ciascuno servito da un file **GGUF** preso dalla
cache di Hugging Face montata in **bind**, e li interroga in sequenza con uno
script Python che ne salva le risposte.

Tutto in **locale**, tramite l'immagine Singularity di `llama.cpp`
(vedi [../../../docs/installazione_container.md](../../../docs/installazione_container.md) e
[../../../docs/uso_llamacpp.md](../../../docs/uso_llamacpp.md)).

## Contenuto

| File | Ruolo |
|------|-------|
| `run_demo_local.sh` | Orchestratore: avvia i tre server in sequenza, lancia il client, arresta tutto al termine. |
| `models.conf` | Definizione delle tre teste: repo HF, file GGUF, porta, contesto, max token. |
| `prompts.txt` | Prompt inviati a ciascun modello (uno per riga). |
| `query_models.py` | Client (solo stdlib): interroga i tre endpoint e salva le risposte. |
| `outputs/` | Risposte salvate (`responses_<timestamp>.json` e `.md`). |
| `logs/` | Log di ciascun `llama-server`. |

## Prerequisiti

1. **Immagine Singularity** di `llama.cpp` costruita in locale
   (`llamacpp_local.sif`) — vedi la guida di installazione. Percorso atteso:
   `../../containers/llamacpp_local.sif`, sovrascrivibile con `CERBERUS_IMAGE`.
2. **Modelli GGUF** presenti nella cache HF. Con la configurazione di default
   servono:
   - `ggml-org/gemma-3-270m-GGUF` (Q8_0);
   - `ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF`;
   - `ggml-org/Qwen3-4B-GGUF` (Q4_K_M):

## Esecuzione

```bash
./run_demo_local.sh
```

Lo script:

1. verifica immagine, cache e configurazione;
2. risolve i percorsi dei `.gguf` nella cache HF;
3. avvia i tre `llama-server` **in sequenza** (attende `/health` prima del
   successivo), sulle porte 8081/8082/8083;
4. lancia `query_models.py`, che invia ogni prompt a ciascun modello e salva le
   risposte in `outputs/`;
5. **arresta automaticamente** tutti i server all'uscita (anche con `Ctrl-C`).

### Parametri sovrascrivibili

```bash
CERBERUS_IMAGE=/percorso/llamacpp-cuda.sif \
HF_HOME=/percorso/cache_hf \
NGL=99 HEALTH_TIMEOUT=180 \
./run_demo_local.sh
```

## Personalizzazione

- **Modelli**: modificare `models.conf` (repo, GGUF, porta, contesto).
- **Prompt**: modificare `prompts.txt`.

> **Nota sulla VRAM.** I tre modelli girano contemporaneamente: con i modelli di
> default (270M + 0.5B + 4B quantizzato) l'occupazione è contenuta. Sostituendoli
> con modelli più grandi, verificare che la somma entri nella VRAM disponibile o
> ridurre `-ngl`/il numero di teste.
