# Cerberus — panoramica: cosa offre e come funziona

Cerberus è uno strumento per **servire uno o più modelli LLM con `llama.cpp` sul
cluster IBiSCo** (SLURM, Singularity, GPU NVIDIA V100-SXM2 da 32 GB, filesystem
Lustre) partendo da **un solo file dichiarativo**. L'utente descrive *quali*
modelli vuole e *quanto* occupano; Cerberus si occupa di tutto il resto:

- **scarica** i file GGUF esatti nella cache di Hugging Face condivisa;
- **decide quante GPU** servono a ciascun modello (in automatico o su indicazione);
- **posiziona** i modelli su nodi e GPU dell'allocazione, impacchettandoli;
- **avvia** i server `llama-server` (uno per modello) con i parametri corretti;
- **genera una mappa degli endpoint** che un client Python usa per instradare le
  richieste al modello giusto, **per etichetta (label)**, tramite l'API
  **compatibile con OpenAI** (`/v1/chat/completions`).

Guide collegate: [esecuzione delle demo](demo.md) ·
[realizzare un progetto](progetto.md) ·
[installazione immagine/sandbox](installazione_container.md).

---

## 1. Il flusso in una figura

```
 models.conf (TOML)                         tu scrivi questo
      │
      │  cerberus validate      → schema OK + quante GPU/nodi servono
      │  cerberus download      → scarica i GGUF nella cache HF
      ▼
 [ allochi i nodi con salloc / sbatch ]      tu allochi le risorse
      │
      │  cerberus up
      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. legge i nodi allocati (scontrol)                       │
 │ 2. stima la VRAM di ogni modello (header GGUF via HTTP)   │
 │ 3. calcola GPU per modello + posizionamento su nodi/GPU   │
 │ 4. lancia 1 `srun` per nodo → avvia i `llama-server`      │
 │ 5. attende che ogni server sia "healthy"                  │
 │ 6. scrive endpoints.json nella cartella del progetto      │
 │ 7. resta in servizio finché non premi Ctrl-C              │
 └──────────────────────────────────────────────────────────┘
      │
      ▼
 endpoints.json  ──►  cerberus.client.py  ──►  il tuo codice di esperimenti
   (label → host:port)     (CerberusClient)      (chat per label)
```

---

## 2. L'input: `models.conf`

Un file **TOML** con una sezione `[allocation]`, una `[defaults]` opzionale e un
blocco `[[model]]` per ciascun modello.

```toml
[allocation]
gpus_per_node = 3          # GPU per nodo disponibili (V100 da 32 GB)

[defaults]                 # valori applicati a ogni [[model]] se non specificati
parallel      = 1
kv_cache_type = "f16"
reasoning     = "auto"

[[model]]
label             = "qwen-big"
hf_repo           = "ggml-org/Qwen3-4B-GGUF"
gguf_file         = "Qwen3-4B-Q4_K_M.gguf"
alloc_mode        = "AUTO"
max_input_tokens  = 8192
max_output_tokens = 2048
reasoning         = "on"
```

### Campi di `[[model]]`

| Campo | Tipo | Obbl. | Significato |
|-------|------|:----:|-------------|
| `label` | str | ✓ | Identificatore unico; il client seleziona il modello con questo |
| `hf_repo` | str | ✓ | Repository HF nella forma `org/repo` (non il nome del file!) |
| `gguf_file` | str | ✓ | Nome esatto del `.gguf` nel repo (la quantizzazione è nel nome). Per i GGUF spezzati, la **prima** parte `…-00001-of-000NN.gguf` |
| `alloc_mode` | `AUTO`\|`MANUAL` | ✓ | Come decidere il numero di GPU |
| `max_input_tokens` | int>0 | ✓ | Token massimi in ingresso attesi |
| `max_output_tokens` | int>0 | ✓ | Token massimi generati attesi |
| `parallel` | int≥1 | – (1) | Slot concorrenti (`--parallel`); incide sul contesto |
| `kv_cache_type` | `f16`\|`q8_0`\|`q4_0` | – (f16) | Quantizzazione della KV-cache (`-ctk`/`-ctv`) |
| `reasoning` | `on`\|`off`\|`auto` | – (auto) | Thinking del modello (`--reasoning`) |
| `reasoning_budget` | int | – | Cap sui token di ragionamento (`--reasoning-budget`) |
| `num_gpus` | int≥1 | ✓ se MANUAL | GPU su cui distribuire il modello (tensor-split) |

Regole di validazione principali: `label` unici; AUTO **non** vuole `num_gpus`
(lo calcola il tool); MANUAL lo richiede e dev'essere ≤ `gpus_per_node` (lo
split non attraversa i nodi).

### Come i token diventano contesto

`llama.cpp` ha un contesto **totale** condiviso tra gli slot: con `--parallel P`
ogni slot riceve `ctx/P`. Cerberus imposta quindi

```
--ctx-size = (max_input_tokens + max_output_tokens) * parallel
--parallel = parallel
```

così **ogni** slot dispone esattamente di `max_input + max_output` token. La
KV-cache viene preallocata a questa dimensione: valori più alti = più VRAM.

---

## 3. AUTO vs MANUAL: quante GPU

### AUTO — deciso dal `cerberus.estimator`
Cerberus stima la VRAM del modello leggendo l'**header del GGUF** (via richieste
HTTP Range, senza scaricare il file) e calcolando:

```
pesi (dimensione su disco del GGUF)
+ KV-cache = 2·layer·(in+out)·parallel·kv_heads·head_dim·byte_per_elemento
+ contesto CUDA  (~0.4 GB per GPU)
+ buffer di calcolo (stima)
+ margine di sicurezza (5% dei pesi)
```

Poi cerca il **numero minimo di GPU** su cui il modello sta entro i 32 GB per GPU
(il contesto CUDA cresce col numero di GPU, quindi è una vera ricerca, non una
divisione):

```
per g = 1, 2, …, gpus_per_node:
    se VRAM_per_gpu(g) ≤ 32 GiB:  num_gpus = g;  stop
```

Esempio: Llama-3.3-70B Q4_K_M (~40 GiB di pesi) non entra in una V100 → `num_gpus = 2`.

### MANUAL — deciso da te
Indichi `num_gpus` e Cerberus distribuisce il modello su quel numero di GPU con
tensor-split. Utile quando vuoi forzare uno split (o riservare capacità).

---

## 4. Il posizionamento (placement)

Dato, per ogni modello, il numero di GPU e l'impronta di VRAM, Cerberus impacchetta
i modelli sull'allocazione `nodi × gpus_per_node` (32 GB per GPU):

- i modelli **multi-GPU** (tensor-split) occupano quel numero di **GPU intere** su
  **un solo nodo** (lo split non può attraversare i nodi);
- i modelli **su singola GPU** vengono impacchettati e possono **condividere** una
  GPU se le loro impronte sommano ≤ 32 GB (first-fit-decreasing);
- i nodi vengono usati man mano; se l'allocazione non basta, `cerberus up` si
  ferma con un errore chiaro (es. *"models do not fit in 1 node(s) x 3 GPU(s)"*).

**Assegnazione automatica di nodi e GPU:** non indichi né il nodo né le GPU — è il
tool a scegliere, conoscendo solo `gpus_per_node`. Gli indici GPU sono *logici*
(relativi all'allocazione SLURM), quindi funzionano qualunque siano le GPU fisiche
assegnate dal job.

---

## 5. L'avvio dei server

Per ogni nodo usato, Cerberus lancia uno step `srun` che esegue il launcher di
nodo; questo avvia i `llama-server` dentro la **sandbox CUDA**, ciascuno su una
**porta libera** (cercata sul nodo, nel range `8081–8999` di default) e in ascolto
su `--host 0.0.0.0` (raggiungibile dagli altri nodi). Il comando per modello:

```
llama-server -m /hf/<gguf> -dev CUDA<a>[,CUDA<b>] \
  -ngl 999 --fit off \
  -c <(in+out)*parallel> -np <parallel> \
  -ctk <kv> -ctv <kv> --jinja \
  --reasoning-format deepseek --reasoning <on|off|auto> [--reasoning-budget N] \
  --host 0.0.0.0 --port <porta-libera> -t <threads> --threads-http <threads_http>
```

Dettagli: `-ngl 999` = offload completo; `--fit off` = niente auto-riduzione
silenziosa (se non entra, errore visibile); `--reasoning-format deepseek` mette
sempre l'eventuale ragionamento in `message.reasoning_content`.

> **Perché la sandbox.** Su IBiSCo non si può costruire un'immagine `.sif` (serve
> root/`--fakeroot`), e i binari di `llama.cpp` sono compilati a parte per sm_70 su
> Lustre. La sandbox fornisce l'ambiente CUDA/glibc giusto; i binari stanno fuori
> (`LD_LIBRARY_PATH`). Vedi [installazione_container.md](installazione_container.md).

---

## 6. L'output: `endpoints.json`

Quando tutti i server sono pronti, Cerberus scrive `endpoints.json` **nella
cartella di `models.conf`**:

```json
{
  "job_id": "1036429",
  "generated_at": "2026-07-05T10:00:00",
  "models": {
    "qwen-big": { "base_url": "http://ibiscohpc-wn27:8123/v1",
                  "host": "ibiscohpc-wn27", "port": 8123, "reasoning": true },
    "coder":    { "base_url": "http://ibiscohpc-wn28:8140/v1",
                  "host": "ibiscohpc-wn28", "port": 8140, "reasoning": false }
  }
}
```

`cerberus up` resta poi in **foreground**, tenendo i server attivi finché non
premi `Ctrl-C` (che li arresta). Il client legge questa mappa per instradare.

---

## 7. Il client

Unico punto d'accesso a tutti i modelli, per label, via API OpenAI:

```python
from cerberus import CerberusClient
c = CerberusClient()                       # legge ./endpoints.json (o $CERBERUS_ENDPOINTS)
r = c.chat("qwen-big", [{"role": "user", "content": "Ciao"}], reasoning=False)
r.content      # risposta pulita (mai <think>)
r.reasoning    # traccia di ragionamento, o None
```

Il client separa il *thinking* dalla risposta (nativo via `reasoning_content`, con
fallback su `<think>…</think>`), ed è **sicuro fuori dal cluster**: importarlo e
costruirlo non tocca file né rete; senza mappa `is_available()` è `False`. Dettagli
d'uso nella [guida al progetto](progetto.md).

---

## 8. Comandi

| Comando | Cosa fa | Dove si lancia |
|---------|---------|----------------|
| `cerberus validate` | valida lo schema, stima GPU/nodi (legge header GGUF via HTTP) | login node (serve rete) |
| `cerberus download` | scarica i GGUF esatti nella cache HF | login node (serve rete) |
| `cerberus up` | posiziona, avvia i server, scrive `endpoints.json`, resta in servizio | shell dell'allocazione (login node) |
| `cerberus status` | mostra UP/DOWN di ogni modello | ovunque veda `endpoints.json` |
| `cerberus down` | rimuove la mappa degli endpoint | — |

`cerberus` equivale a `python -m cerberus`.

---

## 9. Variabili d'ambiente

| Variabile | Default | Scopo |
|-----------|---------|-------|
| `HF_TOKEN` | – | Token Hugging Face (repo gated / download) |
| `HF_HOME` | `~/.cache/huggingface` | Radice della cache HF (meglio su storage condiviso ampio) |
| `CERBERUS_SANDBOX` | `~/tools/cuda-build` | Sandbox CUDA |
| `CERBERUS_BIN` | `~/tools/llama.cpp-master/build/bin` | Binari di `llama.cpp` |
| `CERBERUS_BIND` | `~/tools` | Cartella montata in bind nel container |
| `CERBERUS_THREADS` | `8` | Thread di inferenza per server |
| `CERBERUS_THREADS_HTTP` | `4` | Thread HTTP per server |
| `CERBERUS_PORT_MIN`/`MAX` | `8081`/`8999` | Range delle porte libere |
| `CERBERUS_SRUN_GRES` | (vuoto → `--gpus-per-node=N`) | Sintassi GPU per lo step `srun` (es. `--gres=gpu:3`) |
| `CERBERUS_HEALTH_TIMEOUT` | `600` | Attesa massima (s) per l'avvio dei server |
| `CERBERUS_ENDPOINTS` | – | Percorso esplicito di `endpoints.json` per il client |

---

## 10. Concetti chiave in breve

- **Una sola fonte di verità:** `models.conf`. Guida download, placement, avvio,
  mappa.
- **AUTO calcola le GPU** dal GGUF + token; **MANUAL** le fissi tu.
- **Placement automatico** su nodi e GPU, con condivisione dei piccoli e
  tensor-split intra-nodo dei grandi.
- **Porte libere** scelte a runtime → niente collisioni.
- **Isolamento per progetto:** ogni deploy scrive `endpoints.json` nella propria
  cartella; per esperimenti concorrenti usa cartelle distinte.
- **Reasoning** gestito nativamente e separato dal client.
