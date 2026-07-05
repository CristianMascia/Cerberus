# Cerberus — eseguire le demo

Due demo pronte mostrano Cerberus all'opera su IBiSCo, dalla configurazione fino
alle risposte dei modelli. Stanno in `demo/ibisco/` (singolo nodo) e
`demo/ibisco_multinode/` (due nodi). Prima di iniziare, leggi la
[panoramica](cerberus_panoramica.md); per costruire un tuo progetto vedi la
[guida al progetto](cerberus_progetto.md).

---

## 0. Prerequisiti (una volta sola)

Sul **login node**:

```bash
# ambiente conda del tool
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh
conda create -n cerberus python=3.11 -y
conda activate cerberus

# installazione del tool (comando `cerberus` + client_llamacpp + dipendenze)
cd ~/tools/Cerberus
pip install -e .

# token Hugging Face (mettilo anche nel ~/.bashrc)
export HF_TOKEN=hf_...
```

Devono esistere la **sandbox CUDA** (`~/tools/cuda-build`) e i **binari** di
`llama.cpp` (`~/tools/llama.cpp-master/build/bin`) — vedi
[llamacpp_install.md](llamacpp_install.md). Se sono altrove, esporta
`CERBERUS_SANDBOX` / `CERBERUS_BIN`.

---

## Demo 1 — singolo nodo (`demo/ibisco`)

### In cosa consiste
Serve **due modelli su un solo nodo**:

| Modello | alloc | GPU | Note |
|---------|-------|-----|------|
| `llama33-70b` | AUTO | **2** | Llama-3.3-70B Q4_K_M (~40 GiB): non entra in una V100, AUTO lo splitta su 2 GPU |
| `qwen-coder` | AUTO | 1 | Piccolo, si impacchetta sulla terza GPU |

Totale: **3 GPU → 1 nodo**. Dimostra che l'AUTO decide da solo lo split di un
modello grande e che un modello piccolo condivide il nodo.

### File della demo
| File | Ruolo |
|------|-------|
| `models.conf` | i due modelli, `gpus_per_node = 3` |
| `run_demo.sh` | fa tutto: `download` → `up` (in background) → client → teardown |
| `demo_client.py` | interroga ogni modello via `client_llamacpp`, salva le risposte |
| `prompts.txt` | i prompt inviati |

### Esecuzione
```bash
cd ~/tools/Cerberus/demo/ibisco

# 1) (facoltativo) verifica quante risorse servono
cerberus validate            # → "needs 3 GPU(s) -> 1 node(s)"

# 2) scarica i modelli (la prima volta ~40 GiB per il 70B)
cerberus download

# 3) alloca 1 nodo e lancia (dalla shell dell'allocazione, NON con srun davanti)
salloc --partition=gpus --nodes=1 --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=01:00:00
conda activate cerberus      # riattiva l'env se serve
./run_demo.sh
```

`run_demo.sh` mette su i server, attende `endpoints.json`, esegue `demo_client.py`,
poi arresta tutto. Le risposte finiscono in `outputs/responses_<timestamp>.md` (e
`.json`), i log dei server in `.cerberus/<jobid>/logs/`.

---

## Demo 2 — multi nodo (`demo/ibisco_multinode`)

### In cosa consiste
Serve **quattro modelli distribuiti su due nodi**, con posizionamento automatico:

| Modello | alloc | GPU | Nodo (deciso dal tool) |
|---------|-------|-----|------------------------|
| `llama70-a` | AUTO | 2 | nodo 0 |
| `llama70-b` | AUTO | 2 | nodo 1 |
| `gemma-270m` | AUTO | 1 | impacchettato |
| `qwen-coder` | AUTO | 1 | impacchettato |

Due istanze del 70B (stesso GGUF, scaricato una volta) richiedono 2 GPU intere
ciascuna = **4 GPU > 3 per nodo**, quindi Cerberus le distribuisce su **2 nodi**; i
piccoli si impacchettano nella GPU avanzata di ciascun nodo. Totale: **6 GPU → 2
nodi**. Non indichi nulla: nodo, numero di GPU e split li sceglie il tool.

### File aggiuntivo
`submit.sbatch` — job SLURM (2 nodi × 3 GPU) che lancia `run_demo.sh` in modalità
batch.

### Esecuzione — interattiva
```bash
cd ~/tools/Cerberus/demo/ibisco_multinode
cerberus validate            # → "needs 6 GPU(s) -> 2 node(s)"
cerberus download            # il 70B è scaricato una sola volta

salloc --partition=gpus --nodes=2 --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=01:00:00
conda activate cerberus
./run_demo.sh
```

### Esecuzione — batch
Modifica `submit.sbatch` per attivare l'env e il token **dentro** il job:
```bash
# nello script, prima di 'bash run_demo.sh':
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus
export HF_TOKEN=hf_...
```
poi:
```bash
sbatch submit.sbatch
tail -f logs/slurm-<jobid>.out
```

---

## Cosa aspettarsi (esito corretto)

Nell'output di `cerberus up` (o nel log SLURM) vedi, in ordine:

```
[cerberus] allocation: 2 node(s): ibiscohpc-wn27 ibiscohpc-wn28
[cerberus] computing VRAM needs and placement...
[cerberus] plan uses 2/2 node(s):
  llama70-a: node0 (ibiscohpc-wn27) dev=CUDA0,CUDA1 ctx=4608 (~44.2 GiB)
  llama70-b: node1 (ibiscohpc-wn28) dev=CUDA0,CUDA1 ctx=4608 (~44.2 GiB)
  gemma-270m: node0 (…) dev=CUDA2 …
  qwen-coder: node0 (…) dev=CUDA2 …
[cerberus] launching node 0 (…)
[cerberus] launching node 1 (…)
[cerberus] ready — 4 model(s). endpoints: …/endpoints.json
[cerberus] serving. Press Ctrl-C to stop.
```

Poi `demo_client.py` interroga i modelli e salva `outputs/responses_*.md`. Il
segnale che **funziona**: arrivi a `ready — N model(s)`, `cerberus status` mostra
tutti **UP**, e le risposte nel file non sono vuote.

---

## Errori tipici e come risolverli

| Sintomo | Causa / rimedio |
|---------|-----------------|
| `activate the 'cerberus' env` | `conda activate cerberus` (e `pip install -e .`) |
| `not inside a SLURM allocation` | lancia dalla shell di `salloc`; **non** mettere `srun` davanti a `run_demo.sh` |
| bloccato su `waiting for endpoints.json` dopo `srun ./run_demo.sh` | hai avvolto tutto in `srun` (nodo di calcolo senza rete + `srun` annidato): togli `srun`, lancia `./run_demo.sh` |
| `models do not fit in N node(s)` | hai allocato pochi nodi: usa il numero indicato da `cerberus validate` |
| bloccato su `computing VRAM needs…` | la shell è su un nodo senza rete esterna: esegui `cerberus up` dal login node (che ha internet) |
| timeout "models not healthy" | il 70B su 2 GPU è lento a caricare: `CERBERUS_HEALTH_TIMEOUT=1200 ./run_demo.sh`; oppure OOM/CUDA nel log `.cerberus/<jobid>/logs/<label>.log` |
| errori GPU nei `node_*_srun.log` | sintassi GRES diversa: `CERBERUS_SRUN_GRES="--gres=gpu:3" ./run_demo.sh` |
| download fallisce (repo gated) | `HF_TOKEN` valido con accesso al repo |

---

## Passo-passo per capire dove si rompe

Se qualcosa non va, procedi per gradi (più facile isolare il problema):

```bash
cerberus validate            # 1) schema + stima (no GPU, no allocazione)
cerberus download            # 2) scarica (login node)
# dopo aver allocato:
cerberus up                  # 3) in foreground DA SOLO: vedi placement, porte, "ready"
cerberus status              # 4) UP/DOWN dei modelli
python demo_client.py        # 5) le query
# Ctrl-C sul 'cerberus up' per fermare
```

Se un passo fallisce, guarda il log pertinente in `.cerberus/<jobid>/logs/`
(`node_<i>_srun.log` per problemi di `srun`/GPU, `<label>.log` per l'avvio del
singolo server).
