# Cerberus — guida al tool

Cerberus serve uno o più modelli LLM con `llama.cpp` su IBiSCo a partire da **un
solo file dichiarativo** (`models.conf`, in TOML). Il tool si occupa di: download
dei GGUF, decisione di quante GPU servono, posizionamento su nodi e GPU, avvio dei
server `llama-server` e generazione di una **mappa degli endpoint** che il client
Python usa per instradare le richieste per *label*.

Tutto l'accesso ai modelli avviene tramite l'**endpoint OpenAI-compatibile**
(`/v1/chat/completions`).

## Installazione (ambiente conda `cerberus`)

```bash
conda create -n cerberus python=3.11 -y
conda activate cerberus
pip install -e .                          # installa il comando `cerberus` + le dipendenze
export HF_TOKEN=hf_...                    # la tua chiave HF (per i repo gated)
```

> Il comando `cerberus` equivale a `python -m cerberus`. Se preferisci non
> installare, esegui i comandi come `python -m cerberus …` con la root del repo
> nel `PYTHONPATH`.

Prerequisiti sul cluster (già presenti, vedi [llamacpp_install.md](llamacpp_install.md)):
sandbox CUDA (`~/tools/cuda-build`) e binari di `llama.cpp` (`~/tools/llama.cpp-master/build/bin`).

## 1. Scrivi `models.conf`

```toml
[allocation]
gpus_per_node = 3          # GPU per nodo disponibili (V100 32 GB)

[defaults]
parallel      = 1
kv_cache_type = "f16"
reasoning     = "auto"

[[model]]
label             = "qwen-big"
hf_repo           = "ggml-org/Qwen3-4B-GGUF"
gguf_file         = "Qwen3-4B-Q4_K_M.gguf"   # nome esatto => quantizzazione esplicita
alloc_mode        = "AUTO"                    # il tool calcola le GPU dal GGUF + token
max_input_tokens  = 8192
max_output_tokens = 2048
reasoning         = "on"

[[model]]
label             = "coder"
hf_repo           = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF"
gguf_file         = "qwen2.5-coder-0.5b-q8_0.gguf"
alloc_mode        = "MANUAL"
num_gpus          = 1
max_input_tokens  = 4096
max_output_tokens = 512
```

**Campi.** `label` (identificatore usato dal client), `hf_repo`+`gguf_file` (il
GGUF esatto), `alloc_mode` (`AUTO`/`MANUAL`), `max_input_tokens`/`max_output_tokens`
(dimensionano `--ctx-size = (in+out)·parallel`), `parallel` (slot concorrenti),
`kv_cache_type` (`-ctk/-ctv`), `reasoning` (`on|off|auto` → `--reasoning`),
`reasoning_budget` (opz. → `--reasoning-budget`). In MANUAL si indica `num_gpus`;
in AUTO lo calcola il tool.

## 2. Comandi

```bash
cerberus validate      # controlla lo schema e stima GPU/nodi necessari
cerberus download      # scarica i GGUF esatti nella cache HF condivisa
```

`validate` legge gli header GGUF via HTTP (senza scaricare) e stima la VRAM
(pesi + KV-cache + contesto CUDA + buffer + margine); da lì decide, per i modelli
AUTO, quante GPU servono e quanti nodi allocare.

## 3. Alloca i nodi e avvia

Il numero di nodi suggerito da `validate`:

```bash
salloc --partition=gpus --nodes=<N> --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=02:00:00
cerberus up            # posiziona, avvia i server, scrive endpoints.json, resta in servizio
```

`cerberus up` gira sul master (anche dal login node: orchestra via `srun`), sceglie
una **porta libera** per ogni server, li avvia dentro la sandbox su `--host 0.0.0.0`,
attende l'health e scrive `endpoints.json` **nella cartella di `models.conf`**.
Resta in foreground servendo i modelli finché non premi `Ctrl-C` (che li arresta).

`endpoints.json` (label → `host:port`):
```json
{ "job_id":"…", "models": {
    "qwen-big": {"base_url":"http://ibiscohpc-wn27:8123/v1","reasoning":true}, … } }
```

## 4. Usa i modelli dal client

Da un'altra shell (o dal tuo codice di esperimenti):

```python
from client_llamacpp import CerberusClient

c = CerberusClient()                        # legge ./endpoints.json (o $CERBERUS_ENDPOINTS)
print(c.list_models())

r = c.chat("qwen-big",
           [{"role": "user", "content": "Spiega la ricorsione."}],
           reasoning=False, max_tokens=512)
print(r.content)       # risposta pulita (mai <think>)
print(r.reasoning)     # traccia di ragionamento, o None
```

- `reasoning=True/False/None` chiede/sopprime/lascia-default il thinking (best-effort per richiesta).
- I modelli *thinking* espongono il ragionamento in `reasoning_content` (nativo,
  `--reasoning-format deepseek`); il client lo separa dalla risposta.
- **Off-cluster** il client è sicuro: `import` e costruzione non falliscono;
  senza mappa, `is_available()` è `False`.

Stato e arresto:
```bash
cerberus status        # UP/DOWN di ogni modello
# per fermare: Ctrl-C sul 'cerberus up', oppure scancel del job
```

## Note

- **Due job insieme:** ogni deploy scrive `endpoints.json` nella **propria**
  cartella di progetto (quella di `models.conf`); usa cartelle distinte per
  esperimenti concorrenti. Un solo `cerberus up` per cartella. Le porte sono
  scelte libere sul nodo, quindi niente collisioni anche su nodi condivisi.
- **AUTO vs MANUAL:** AUTO calcola le GPU con il `vram_estimator` (che considera
  già KV-cache, contesto CUDA, buffer e margine) e confronta con i 32 GB della
  V100; MANUAL usa il numero che indichi (lo split è sempre intra-nodo).
- **Placement automatico:** modelli piccoli possono condividere una GPU; modelli
  grandi vengono distribuiti su più GPU dello **stesso** nodo (tensor-split).
