# Cerberus — demo su IBiSCo (tre teste LLM, multi-GPU)

Variante per **IBiSCo** della [demo locale](../local/README.md): avvia **tre
modelli LLM** con `llama.cpp` (le tre teste di Cerberus) usando la **sandbox
CUDA** e i binari compilati su **Lustre** (vedi
[../../guides/llamacpp_install.md](../../guides/llamacpp_install.md), Parte 2), e
li interroga in sequenza salvandone le risposte.

Rispetto alla demo locale cambiano due cose:

- **avvio** tramite `singularity exec` sulla sandbox CUDA (`~/tools/cuda-build/`)
  con `LD_LIBRARY_PATH` sui binari di llama.cpp (`~/tools/llama.cpp-master/build/bin/`)
  e `--host 0.0.0.0`;
- **assegnazione delle GPU**: il nodo ha più GPU e ogni testa va posta su una GPU
  specifica, con due modalità (`manual` / `auto`).

## Contenuto

| File | Ruolo |
|------|-------|
| `run_demo_ibisco.sh` | Orchestratore: assegna le GPU, avvia i tre server in sequenza, lancia il client, arresta tutto al termine. |
| `allocate_gpus.py` | Assegna una GPU a ciascun modello, in modalità `manual` o `auto`. |
| `models.conf` | Le tre teste: repo HF, GGUF, porta, contesto, max token, **cuda_device**, **est_mem_mb**. |
| `prompts.txt` | Prompt inviati a ciascun modello. |
| `query_models.py` | Client (solo stdlib): interroga i tre endpoint e salva le risposte. |
| `outputs/`, `logs/` | Risposte salvate e log dei server. |

## Le due modalità di assegnazione GPU

Si sceglie con `CERBERUS_MODE` (default: `manual`).

### `manual` — GPU indicata in `models.conf`

Ogni modello usa la GPU del campo `cuda_device`. **Gli indici sono logici**,
cioè relativi alle sole GPU assegnate dal job SLURM, non ai numeri fisici delle
schede.

Quando si sottomette un job non si conosce in anticipo quali GPU fisiche saranno
assegnate: richiedendo 3 GPU si potrebbero ottenere, ad esempio, le fisiche
`0,2,3`. SLURM le espone tramite `CUDA_VISIBLE_DEVICES=0,2,3`, e `llama.cpp` le
vede come indici logici `CUDA0,CUDA1,CUDA2`. Perciò un `cuda_device` pari a
`0,1,2` in `models.conf` mappa automaticamente:

| indice logico (`models.conf`) | GPU fisica (esempio allocazione `0,2,3`) |
|:-----------------------------:|:----------------------------------------:|
| 0 | 0 |
| 1 | 2 |
| 2 | 3 |

Non occorre quindi conoscere i numeri fisici: si ragiona sempre in termini di
"prima, seconda, terza GPU dell'allocazione". Per uno **split multi-GPU** di un
modello grande si usa una lista, es. `cuda_device = 1,3` → `--device CUDA1,CUDA3`.

### `auto` — allocazione automatica per memoria stimata

Il campo `cuda_device` è ignorato; si usa `est_mem_mb` (memoria stimata di
ciascun modello). Lo script interroga la memoria **libera** di ogni GPU
dell'allocazione (`nvidia-smi`) e colloca i modelli con un **first-fit**: più
modelli piccoli possono condividere una GPU, un modello grande ne ottiene una
capiente. Viene lasciato un margine di sicurezza di 512 MB per GPU.

**Modelli oltre la singola GPU (split multi-GPU).** Un modello la cui memoria
stimata eccede quella di ogni singola GPU (ad esempio oltre i 32 GB di una V100)
viene **distribuito automaticamente su più GPU**: lo script sceglie un insieme di
GPU la cui memoria libera combinata è sufficiente ed emette un `--tensor-split`
proporzionale alla capienza di ciascuna, così che nessuna vada in overflow. Lo
stesso in modalità `manual` è ottenibile elencando le GPU in `cuda_device` (es.
`0,1`), con ripartizione uniforme di default. Se nemmeno l'insieme delle GPU
disponibili basta, l'allocazione fallisce con messaggio esplicito.

## Esecuzione

Da un **nodo di calcolo con GPU**, tipicamente dentro un job SLURM:

```bash
# allocazione interattiva (adattare partizione e numero di GPU)
salloc --partition=<gpu_part> --gres=gpu:3 --time=02:00:00

# modalità manuale (default): GPU da models.conf
./run_demo_ibisco.sh

# oppure allocazione automatica per memoria
CERBERUS_MODE=auto ./run_demo_ibisco.sh
```

Lo script:

1. verifica sandbox, binari, cache e configurazione;
2. assegna le GPU (`allocate_gpus.py`, modalità scelta);
3. risolve i `.gguf` nella cache HF;
4. avvia i tre `llama-server` **in sequenza** (attende `/health`), con
   `--device CUDA<n>` e `--host 0.0.0.0`, sulle porte 8081/8082/8083;
5. lancia `query_models.py` (che si collega a `127.0.0.1` sullo stesso nodo) e
   salva le risposte in `outputs/`;
6. **arresta automaticamente** tutti i server all'uscita (anche con `Ctrl-C`).

### Parametri sovrascrivibili

```bash
CERBERUS_MODE=auto \
CERBERUS_SANDBOX=$HOME/tools/cuda-build \
CERBERUS_BIN=$HOME/tools/llama.cpp-master/build/bin \
CERBERUS_BIND=$HOME/tools \
HF_HOME=/ibiscostorage/$USER/hf_cache \
NGL=99 HEALTH_TIMEOUT=300 \
./run_demo_ibisco.sh
```

### Accesso da remoto

Il client gira sullo stesso nodo dei server (`127.0.0.1`). Per raggiungere le
interfacce web/API dalla propria macchina, aprire un tunnel SSH attraverso il
login node (vedi [../../guides/llamacpp_usage.md](../../guides/llamacpp_usage.md),
§ 2.4):

```bash
ssh -L 8081:<nodo>:8081 -L 8082:<nodo>:8082 -L 8083:<nodo>:8083 <utente>@<login-node>
```

> **Nota.** Il file `.gguf` di Qwen3-4B deve essere presente nella cache HF; se
> manca, scaricarlo (vedi [../../guides/llm_download.md](../../guides/llm_download.md)):
> `hf download ggml-org/Qwen3-4B-GGUF --include "*Q4_K_M*"`.
