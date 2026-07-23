# Cerberus — demo locale (richieste parallele vs sequenziali)

Esempio di progetto che avvia **un modello LLM** con `llama.cpp` su Singularity,
servito da un file **GGUF** preso dalla cache di Hugging Face montata in
**bind**, e lo interroga con uno script Python che invia le stesse richieste
prima **in sequenza** e poi **tutte in parallelo**, misurando e confrontando i
due tempi totali.

Tutto in **locale**, tramite l'immagine Singularity di `llama.cpp`
(vedi [../../../docs/installazione_container.md](../../../docs/installazione_container.md) e
[../../../docs/uso_llamacpp.md](../../../docs/uso_llamacpp.md)).

## Contenuto

| File | Ruolo |
|------|-------|
| `run_demo_local.sh` | Orchestratore: avvia il server, lancia il client, arresta tutto al termine. |
| `models.conf` | Definizione del modello: repo HF, file GGUF, porta, contesto, max token. |
| `prompts.txt` | Prompt inviati al modello (uno per riga); formulati per ottenere risposte lunghe. |
| `query_models.py` | Client (solo stdlib): invia i prompt in sequenza e poi in parallelo, salva le risposte e i tempi. |
| `outputs/` | Risposte e timing salvati (`responses_<timestamp>.json` e `.md`). |
| `logs/` | Log del `llama-server`. |

## Prerequisiti

1. **Immagine Singularity** di `llama.cpp` costruita in locale
   (`llamacpp_local.sif`) — vedi la guida di installazione. Percorso atteso:
   `../../containers/llamacpp_local.sif`, sovrascrivibile con `CERBERUS_IMAGE`.
2. **Modello GGUF** presente nella cache HF. Con la configurazione di default
   serve `ggml-org/gemma-3-270m-GGUF` (Q8_0).

## Esecuzione

```bash
./run_demo_local.sh
```

Lo script:

1. verifica immagine, cache e configurazione;
2. risolve il percorso del `.gguf` nella cache HF;
3. avvia `llama-server` con `--parallel "$NPARALLEL"` (slot di inferenza
   concorrenti) e attende `/health` prima di procedere;
4. lancia `query_models.py`, che invia tutti i prompt di `prompts.txt` prima
   **uno alla volta** (sequenziale) e poi **tutti insieme** con un thread pool
   (parallelo), misura il tempo totale di ciascuna modalità e salva risposte +
   timing in `outputs/`;
5. **arresta automaticamente** il server all'uscita (anche con `Ctrl-C`).

### Parametri sovrascrivibili

```bash
CERBERUS_IMAGE=/percorso/llamacpp-cuda.sif \
HF_HOME=/percorso/cache_hf \
NGL=99 NPARALLEL=20 HEALTH_TIMEOUT=180 \
./run_demo_local.sh
```

`NPARALLEL` (`-np`/`--parallel` di `llama-server`) è il numero di slot di
inferenza concorrenti del server: deve coprire il numero di richieste che
`query_models.py` invia in parallelo (di default una per ogni riga di
`prompts.txt`, quindi `NPARALLEL` di default è 20), altrimenti il server le
mette comunque in coda anche se il client le invia tutte insieme. Il
contesto (`-c` in `models.conf`) viene diviso tra gli slot, quindi ogni slot
ha una finestra pari a `ctx/NPARALLEL`: aumentando `NPARALLEL` va aumentato
anche il contesto in `models.conf` per non restare senza spazio per
prompt + risposta.

## Personalizzazione

- **Modello**: modificare `models.conf` (repo, GGUF, porta, contesto, max token).
- **Prompt**: modificare `prompts.txt`. Numero di righe = numero di richieste
  inviate in parallelo nel secondo run.

## Nota sullo speedup

Non aspettarti uno speedup lineare (es. 20 richieste parallele ≠ 20× più
veloce). Gli slot paralleli di `llama-server` condividono lo stesso motore di
calcolo (GPU o CPU): non aggiungono potenza di calcolo, ne migliorano solo
l'utilizzo tramite continuous batching. Una singola richiesta sequenziale già
usa buona parte della GPU/CPU disponibile; il batching aiuta soprattutto
quando la fase di decode è *memory-bandwidth-bound* (i pesi vengono letti una
volta e riusati per più sequenze), ma oltre una certa dimensione del batch il
collo di bottiglia diventa il calcolo puro (FLOPs) e lo speedup satura — con
modelli piccoli come `gemma-3-270m` è normale osservare speedup nell'ordine
di 3-5× anche con `NPARALLEL` più alto, non 15-20×.

> **Nota sulla VRAM.** Con `NPARALLEL` alto e contesto proporzionalmente
> alzato in `models.conf`, verificare che la VRAM disponibile sia sufficiente
> (KV cache ∝ contesto totale) o ridurre `NGL`/`NPARALLEL`.
