# Cerberus — realizzare un progetto che sfrutta Cerberus

Guida pratica per costruire un tuo progetto (esperimenti, pipeline, valutazioni)
che usa uno o più modelli serviti da Cerberus. Copre tutto il ciclo: struttura del
progetto, scrittura di `models.conf`, allocazione delle risorse, avvio dei modelli
e **integrazione del client `llama.cpp` nel tuo codice**.

Prerequisiti concettuali: [panoramica](panoramica.md). Per vedere esempi
funzionanti: [esecuzione delle demo](demo.md).

---

## 1. Struttura consigliata del progetto

Ogni progetto è **una cartella con il suo `models.conf`**. Questo è importante:
`cerberus up` scrive `endpoints.json` nella cartella di `models.conf`, e
l'isolamento tra esperimenti concorrenti è **per cartella**.

```
mio-progetto/
├── models.conf          # quali modelli servire (lo scrivi tu)
├── experiment.py        # il tuo codice, usa cerberus.client
├── run.sbatch           # (opzionale) job che avvia i modelli ed esegue l'esperimento
├── prompts/ dati/ …     # i tuoi input
├── outputs/             # i tuoi risultati
├── endpoints.json       # GENERATO da 'cerberus up'   (da .gitignore)
└── .cerberus/           # runtime interno del tool     (da .gitignore)
```

`.gitignore` consigliato:
```
endpoints.json
.cerberus/
outputs/
__pycache__/
```

---

## 2. Installazione dell'ambiente (una volta)

Sul **login node**:
```bash
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh
conda create -n cerberus python=3.11 -y
conda activate cerberus
cd ~/tools/Cerberus && pip install -e .     # comando `cerberus` + modulo cerberus.client
export HF_TOKEN=hf_...                       # nel ~/.bashrc, se possibile
```

Dopo `pip install -e .`, dal tuo codice funziona `from cerberus import
CerberusClient` in qualsiasi cartella (il modulo è installato nell'env).

---

## 3. Scrivere `models.conf`

Parti dall'esempio annotato `models.conf.example` nella root del repo. Struttura
minima:

```toml
[allocation]
gpus_per_node = 3          # deve corrispondere alle GPU/nodo che allocherai

[defaults]
parallel      = 1
kv_cache_type = "f16"
reasoning     = "auto"

[[model]]
label             = "assistant"
hf_repo           = "ggml-org/Qwen3-4B-GGUF"
gguf_file         = "Qwen3-4B-Q4_K_M.gguf"
alloc_mode        = "AUTO"
max_input_tokens  = 8192
max_output_tokens = 2048
reasoning         = "on"

[[model]]
label             = "big"
hf_repo           = "lmstudio-community/Llama-3.3-70B-Instruct-GGUF"
gguf_file         = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
alloc_mode        = "AUTO"           # → 2 GPU in automatico (non entra in una V100)
max_input_tokens  = 4096
max_output_tokens = 1024
```

### Come scegliere i campi

- **`hf_repo` + `gguf_file`**: `hf_repo` è `org/repo` (NON il nome del file);
  `gguf_file` è il nome esatto del `.gguf` in quel repo. Per trovarlo, apri la
  scheda *Files* del modello su Hugging Face, oppure usa lo stimatore:
  `python cerberus/estimator/gguf_vram.py <org/repo> -q Q4_K_M`. Per i GGUF **spezzati**
  indica la prima parte (`…-00001-of-000NN.gguf`): Cerberus scarica tutte le parti.
- **`max_input_tokens` / `max_output_tokens`**: la finestra che vuoi garantire per
  richiesta. Determina `--ctx-size = (in+out)·parallel` e quindi la VRAM della
  KV-cache. Non esagerare: valori alti = più memoria e, in AUTO, più GPU.
- **`parallel`**: quante richieste concorrenti vuoi servire davvero in parallelo.
  Ogni slot riceve `in+out` token, quindi `parallel=4` quadruplica la KV-cache.
- **`alloc_mode`**: `AUTO` (consigliato) fa calcolare le GPU al tool; `MANUAL` +
  `num_gpus` se vuoi forzare uno split.
- **`reasoning`**: `on`/`off`/`auto`. Per modelli *thinking* (es. Qwen3) `on` fa
  emettere il ragionamento (separato nel client); per modelli instruct normali
  (es. Llama 3.3) lascia `auto`.
- **`kv_cache_type`**: `q8_0` dimezza (circa) la VRAM della KV-cache rispetto a
  `f16`, utile per contesti lunghi, con impatto minimo sulla qualità.

### Validare prima di allocare
```bash
cd mio-progetto
cerberus validate
```
Stampa, per ogni modello, `num_gpus` e VRAM stimata, e soprattutto **quanti nodi
allocare**:
```
[cerberus] needs 3 GPU(s) -> 1 node(s) at 3 GPU/node.
[cerberus] allocate e.g.: salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=3 ...
```

---

## 4. Scaricare i modelli
Sul **login node** (unico con rete esterna):
```bash
cerberus download            # scarica i GGUF esatti nella cache HF condivisa
```
Idempotente: i file già in cache non vengono riscaricati. Per repo *gated* serve
`HF_TOKEN` con accesso.

> Suggerimento: se la home ha quota ridotta, punta la cache su storage ampio:
> `export HF_HOME=/ibiscostorage/$USER/hf_cache` (prima di `download` e `up`).

---

## 5. Allocare le risorse e avviare i modelli

Usa il numero di nodi indicato da `validate`. **Regole d'oro:**

- alloca con `--gpus-per-node=<gpus_per_node del tuo models.conf>`;
- lancia `cerberus up` **dalla shell dell'allocazione** (di norma il login node,
  che ha rete per la stima VRAM e può usare `srun`);
- **non** mettere `srun` davanti a `cerberus up`/`run.sh`: è Cerberus a fare gli
  `srun` verso i nodi. Avvolgerlo in `srun` lo esegue su un nodo di calcolo (senza
  rete esterna) e con `srun` annidato → si blocca.

### Modalità A — interattiva (due shell)
```bash
# shell 1: alloca e avvia i server (resta in servizio)
salloc --partition=gpus --nodes=1 --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=04:00:00
conda activate cerberus
cd mio-progetto
cerberus up                       # stampa il piano, poi "ready — N model(s)"

# shell 2 (login node o dentro l'allocazione): usa i modelli
cd mio-progetto
cerberus status                   # UP/DOWN
python experiment.py              # il tuo codice
```
Per fermare: `Ctrl-C` nella shell 1 (arresta tutti i server).

### Modalità B — batch (un solo job fa tutto)
`run.sbatch`:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=mio-progetto
#SBATCH --partition=gpus
#SBATCH --nodes=1                 # <-- da 'cerberus validate'
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus
export HF_TOKEN=hf_...
DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"     # SLURM esegue una copia dello script: usa questa
cd "$DIR"
mkdir -p logs

rm -f endpoints.json
cerberus up &                          # avvia i server in background
UP=$!
trap 'kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true' EXIT

echo "attendo endpoints.json..."
for _ in $(seq 1 400); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "server non partiti (vedi .cerberus/*/logs)"; exit 1; }

python experiment.py                   # il tuo esperimento
# all'uscita, il trap ferma i server
```
```bash
sbatch run.sbatch
```

---

## 6. Integrare il client nel tuo codice

Il modulo è `cerberus.client` (installato con `pip install -e .`). Tutto passa per
la classe `CerberusClient`.

### Le basi
```python
from cerberus import CerberusClient

# Legge endpoints.json: da CERBERUS_ENDPOINTS, oppure da project_dir/endpoints.json,
# oppure da ./endpoints.json. Se il tuo script gira nella cartella del progetto,
# CerberusClient() basta; altrimenti passa project_dir o imposta CERBERUS_ENDPOINTS.
c = CerberusClient()                      # oppure CerberusClient(project_dir="/path/mio-progetto")

print(c.list_models())                    # ['assistant', 'big']

resp = c.chat(
    "assistant",                          # quale modello (label)
    [{"role": "user", "content": "Spiega la ricorsione in una frase."}],
    max_tokens=256,
    temperature=0.7,
)
print(resp.content)                       # risposta pulita, senza <think>
```

### L'oggetto `Response`
| Attributo | Contenuto |
|-----------|-----------|
| `.content` | risposta finale pulita (mai `<think>…</think>`) |
| `.reasoning` | traccia di ragionamento (stringa) o `None` |
| `.finish_reason` | `"stop"` o `"length"` (troncato per `max_tokens`) |
| `.raw` | l'oggetto risposta grezzo dell'SDK OpenAI |

### Modelli con reasoning
```python
r = c.chat("assistant",
           [{"role": "user", "content": "Un treno... a che ora arriva?"}],
           reasoning=True,          # None = default del server, True = pensa, False = non pensare
           max_tokens=1024)         # lascia spazio: thinking + risposta
if r.reasoning:
    print("RAGIONAMENTO:", r.reasoning)
print("RISPOSTA:", r.content)
```
- `reasoning=True/False` chiede/sopprime il thinking per quella richiesta
  (best-effort, via `chat_template_kwargs`); `None` lascia il default del server
  (dato da `reasoning` in `models.conf`).
- Se `.content` è vuoto e `.finish_reason == "length"`, il modello ha esaurito i
  token dentro il ragionamento: alza `max_output_tokens`/`max_tokens` o metti
  `reasoning = "off"`.

### Instradare su più modelli
```python
domanda = [{"role": "user", "content": "Riassumi questo testo: ..."}]
veloce = c.chat("assistant", domanda, max_tokens=300).content   # 4B
accurato = c.chat("big", domanda, max_tokens=600).content       # 70B
```
Il client sceglie automaticamente `host:port` giusto per ogni label.

### Streaming e parametri extra
```python
stream = c.chat("assistant", domanda, stream=True)   # ritorna lo stream dell'SDK OpenAI
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# parametri OpenAI aggiuntivi passano direttamente:
c.chat("assistant", domanda, top_p=0.9, stop=["\n\n"], extra_body={"repeat_penalty": 1.1})
```

### Sviluppo fuori dal cluster (sicurezza)
Importare e costruire il client **non** tocca file né rete: puoi scrivere e
importare il tuo codice sul portatile. Senza mappa degli endpoint:
```python
c = CerberusClient()
if not c.is_available():
    print("nessun modello attivo (esegui su IBiSCo dopo 'cerberus up')")
else:
    ...
```
Una chiamata `chat()` senza mappa solleva `CerberusUnavailable`.

### Un `experiment.py` completo
```python
#!/usr/bin/env python3
import json, time
from pathlib import Path
from cerberus import CerberusClient

HERE = Path(__file__).resolve().parent
PROMPTS = ["Cos'è un GGUF?", "Scrivi una funzione fattoriale in Python.",
           "Tre vantaggi dell'inferenza locale."]

def main():
    c = CerberusClient(project_dir=str(HERE))
    if not c.is_available():
        raise SystemExit("endpoints.json assente: esegui 'cerberus up' prima.")

    results = []
    for label in c.list_models():
        for p in PROMPTS:
            t0 = time.perf_counter()
            r = c.chat(label, [{"role": "user", "content": p}], max_tokens=512)
            results.append({
                "model": label, "prompt": p,
                "answer": r.content, "reasoning": r.reasoning,
                "finish_reason": r.finish_reason,
                "latency_s": round(time.perf_counter() - t0, 2),
            })
            print(f"[{label}] {p[:30]}… -> {r.finish_reason}")

    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"salvato {out/'results.json'}")

if __name__ == "__main__":
    main()
```

---

## 7. Dove far girare il codice del client

Il client parla ai server via `http://<nodo>:<porta>/v1` (i server sono su
`0.0.0.0`). Quindi `experiment.py` può girare:

- sul **login node** (raggiunge i nodi di calcolo sulla rete del cluster) — comodo
  per esperimenti interattivi mentre i server restano su;
- **dentro l'allocazione** (modalità batch di §5) — tutto in un job.

Se lo lanci da una cartella diversa da quella di `models.conf`, indica dov'è la
mappa:
```bash
export CERBERUS_ENDPOINTS=/path/mio-progetto/endpoints.json
python altrove/experiment.py
```
o in Python: `CerberusClient(project_dir="/path/mio-progetto")`.

Per usare i modelli **dal tuo portatile** via tunnel SSH (interfaccia web/API),
vedi §2.4 di [uso_llamacpp.md](uso_llamacpp.md).

---

## 8. Più progetti / più job in contemporanea

Ogni deploy scrive `endpoints.json` nella **propria cartella di progetto**, e le
porte sono scelte libere sul nodo. Quindi due esperimenti concorrenti sono isolati
**se usano cartelle diverse**:

```
progetto-A/models.conf → progetto-A/endpoints.json   (job 1)
progetto-B/models.conf → progetto-B/endpoints.json   (job 2)
```
Regola: **un solo `cerberus up` per cartella** alla volta (un secondo `up` dalla
stessa cartella sovrascriverebbe la mappa). Il client di ogni progetto legge il
proprio `endpoints.json`.

---

## 9. Chiudere

- Interattivo: `Ctrl-C` sul `cerberus up` (arresta i server); oppure `scancel <job>`.
- Batch: al termine dello script il `trap` ferma i server; o `scancel`.
- `cerberus down` rimuove `endpoints.json` (mappa stantia) ma non ferma i server.

---

## 10. Checklist end-to-end

```bash
# 0. una volta: env
conda activate cerberus                      # (pip install -e . fatto)
export HF_TOKEN=hf_...

# 1. progetto
mkdir mio-progetto && cd mio-progetto
$EDITOR models.conf                          # scrivi i modelli
cerberus validate                            # schema + quanti nodi

# 2. modelli
cerberus download

# 3. risorse + avvio
salloc --partition=gpus --nodes=<N> --ntasks-per-node=1 --gpus-per-node=<G> \
       --cpus-per-task=8 --time=04:00:00
cerberus up                                  # in una shell; resta in servizio

# 4. esperimento (altra shell / login node)
python experiment.py                         # usa CerberusClient

# 5. fine
#   Ctrl-C sul 'cerberus up'
```

## 11. Errori frequenti (rimando)
La tabella completa di diagnosi è in
[esecuzione delle demo](demo.md#errori-tipici-e-come-risolverli):
in sintesi — non avvolgere `up` in `srun`; alloca i nodi indicati da `validate`;
esegui `download`/`up` dove c'è rete; per il thinking lascia budget di token; se
`srun` si lamenta delle GPU usa `CERBERUS_SRUN_GRES="--gres=gpu:N"`.
