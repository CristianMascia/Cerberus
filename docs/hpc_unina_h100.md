# Cerberus sull'HPC UniNA (2x H100 per nodo)

Cerberus gira sull'HPC UniNA (cluster `unina-hpc`) con lo stesso codice di
IBiSCo. Cambiano tre cose: la VRAM per GPU nel `models.conf`, l'immagine di
llama.cpp (una `.sif` compilata per sm_90) e il profilo d'ambiente
`profiles/unina-h100.env`.

## Il cluster in breve

| | IBiSCo | UniNA (`ingnormal`) |
| --- | --- | --- |
| GPU per nodo | 3-4x V100-SXM2 32 GB (sm_70) | 2x H100 NVL 94 GB (sm_90) |
| Collegamento tra GPU | NVLink | solo PCIe, su socket diversi |
| Nodi | partizione `gpus` | `r7625ing-gpu[01-08]` (anche `normal`, `dc1normal`) |
| CPU / RAM per nodo | | 256 / 773 GB (ma 10 CPU per job, vedi sotto) |
| Container | Singularity, sandbox + binari in `~/tools` | Apptainer 1.4, `.sif` con llama-server dentro |
| Home | Lustre | Lustre (`/lustre/home/$USER`, senza quota) |
| Python | conda | Python 3.9 di sistema, niente moduli: serve Miniforge |

L'account SLURM è `allusers` ed è quello di default, quindi non serve `--account`.
Non c'è limite di tempo sulle partizioni, ma ogni partizione ha una QOS con
limiti **per job** (massimo 20 job per utente ovunque):

| Partizione | Nodi | CPU per job | GPU per job |
| --- | --- | --- | --- |
| `ingnormal` | 8 | 10 | 4 (cioè al massimo 2 nodi) |
| `dc1normal` | 13 | 13 | 4 |
| `normal` | 6 | 50 | 40 |

Il limite di CPU vale per l'intero job, non per nodo: un job su 2 nodi di
`ingnormal` ha 5 CPU per nodo. Nella batch API ogni shard è un job a sé, quindi
ogni shard ha i suoi 10 CPU e 4 GPU. La partizione di default è `normal`:
specifica sempre `-p ingnormal`.

Il login node (`uninahpc-ui`) non ha GPU, `apptainer`, `git` né `cmake`: va
bene per `cerberus validate`, `download` e `batch submit` (e per il clone, con
il `git` di Miniforge), mentre build delle immagini e test con le GPU si fanno
su un nodo di calcolo.

## Layout delle cartelle

Tutto sta sotto la home su Lustre, condivisa tra login e nodi. Le immagini dei
due backend (llama.cpp ora, vLLM in seguito) stanno insieme in `~/containers`,
e la cache HF è una sola per GGUF e safetensors.

```
~/miniforge3/              conda: env "cerberus" (e in seguito "vllm-base")
~/Projects/Cerberus/       questa repo (main), installata con pip install -e
~/containers/              llamacpp_h100.sif (e in seguito l'immagine vLLM)
~/.cache/huggingface/      cache HF unica per entrambi i backend
~/.apptainer/cache/        cache di Apptainer, fuori da /tmp (70 GB condivisi)
```

## 1. Python e repo (una volta, sul login node)

```bash
curl -L -o Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p $HOME/miniforge3 && rm Miniforge3.sh
$HOME/miniforge3/bin/conda install -n base -y git    # sul login git non c'è
$HOME/miniforge3/bin/conda init bash
# la shell di login (ssh) legge ~/.bash_profile: deve richiamare ~/.bashrc
grep -q bashrc ~/.bash_profile 2>/dev/null || echo '[ -f ~/.bashrc ] && . ~/.bashrc' >> ~/.bash_profile
exec bash -l

mkdir -p ~/Projects ~/containers ~/.apptainer/cache ~/.cache/huggingface/hub
# pesi e immagini sono file da GB: distribuiti su 4 OST si leggono più in fretta
lfs setstripe -c 4 ~/.cache/huggingface/hub
lfs setstripe -c 4 ~/containers

git clone https://github.com/CristianMascia/Cerberus.git ~/Projects/Cerberus
cd ~/Projects/Cerberus
conda env create -f environment.yml && conda activate cerberus   # include git
pip install -e .
python -m unittest discover -s test/unit

# profilo sempre attivo, anche dentro srun e sbatch
echo 'source ~/Projects/Cerberus/profiles/unina-h100.env' >> ~/.bashrc
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc
```

`PYTHONNOUSERSITE=1` evita che pacchetti installati con `pip install --user` in
`~/.local` finiscano negli env conda e dentro i container.

## 2. Verifica del nodo e immagine llama.cpp (una volta, su un nodo GPU)

Il login node non ha `apptainer`: serve una shell **su** un nodo di calcolo
(`srun --pty`; con `salloc` la shell può restare sul login). Chiedere entrambe le
GPU tiene altri job fuori dal nodo, perché le GPU non sono isolate tra job.

```bash
srun -p ingnormal -N1 --gpus-per-node=2 --cpus-per-task=10 --mem=200G \
     --time=03:00:00 --pty bash
cd ~/Projects/Cerberus
bash scripts/check_hpc.sh --vllm      # driver, apptainer, fakeroot, rete; poi prova vLLM
CUDA_ARCH=90 cerberus/containers/build_llamacpp_local.sh \
    llamacpp_local.def ~/containers/llamacpp_h100.sif
```

`check_hpc.sh` scrive un report `hpc_check_<nodo>_<data>.txt` nella cartella
corrente. Con `--vllm` scarica anche l'immagine ufficiale `vllm/vllm-openai:v0.32.0`
(~10 GB, in `~/cerberus-check`) e prova `vllm serve` con un modello piccolo, su
una GPU e su due: le righe `VLLM_TP1_OK` e `VLLM_TP2_OK` dicono che il nodo è
pronto per il backend vLLM.

La build con `--fakeroot` funziona anche senza voce in `/etc/subuid`.
`CUDA_ARCH=90` fissa l'architettura delle H100 (sm_90). I warning
`xattr ... ENOTSUP` durante la build sono dovuti a Lustre e sono innocui. Il
driver 590 (CUDA 13.1) esegue senza problemi l'immagine basata su CUDA 12.4.

## 3. Configurazione

```bash
source ~/Projects/Cerberus/profiles/unina-h100.env   # già nel .bashrc: CERBERUS_IMAGE, HF_HOME, thread
```

Nel `models.conf`:

```toml
[allocation]
gpus_per_node = 2
vram_gib      = 92      # GiB utilizzabili di UNA GPU (default 32 = V100)

[batch]
partition     = "ingnormal"
cpus_per_task = 10
```

Con 92 GiB per GPU il posizionamento automatico cambia: un 70B Q4_K_M
(~40 GiB) che su IBiSCo veniva diviso su 2 GPU qui ne usa una sola, e più
modelli medi condividono la stessa GPU. Dividere un modello su 2 GPU (`num_gpus = 2`
o AUTO quando non sta su una) funziona, ma senza NVLink è più lento: conviene
solo se il modello non entra in 92 GiB.

## 4. Avvio

`--gpus-per-node` va sempre chiesto: senza, il job non ha GPU e gli `srun` di
`cerberus up` restano in coda.

```bash
cerberus validate && cerberus download
salloc -p ingnormal -N1 --ntasks-per-node=1 --gpus-per-node=2 \
       --cpus-per-task=10 --mem=200G --time=04:00:00
cerberus up
```

La batch API (`cerberus batch submit`) usa `partition` e `cpus_per_task` della
sezione `[batch]`. La demo completa è in `test/demo/unina_h100/`.

## 5. Backend vLLM

Accanto a llama.cpp, un modello può essere servito da vLLM con
`backend = "vllm"` nel suo `[[model]]`. Cerberus lancia `vllm serve` dentro
l'immagine ufficiale, sullo snapshot safetensors scaricato da `cerberus download`.
Endpoints, client e batch API restano gli stessi, e i due backend si possono
mescolare nello stesso `cerberus up` o batch.

L'immagine è quella ufficiale, senza build. Se hai già lanciato
`check_hpc.sh --vllm`, basta spostare quella:

```bash
mv ~/cerberus-check/vllm_v0.32.0.sif ~/containers/vllm-0.32.0.sif
# oppure, su un nodo di calcolo:
# apptainer pull ~/containers/vllm-0.32.0.sif docker://vllm/vllm-openai:v0.32.0
```

Il profilo la cerca in `CERBERUS_VLLM_IMAGE` (default
`~/containers/vllm-0.32.0.sif`). La v0.32.0 usa CUDA 13.0, che il driver 590
esegue; da evitare le nightly `cu134-*`.

```toml
[[model]]
label             = "qwen3-8b"
backend           = "vllm"
hf_repo           = "Qwen/Qwen3-8B"     # safetensors, non GGUF (niente gguf_file)
# revision        = "<commit>"          # opzionale: fissa i pesi
alloc_mode        = "AUTO"              # o MANUAL con num_gpus = tensor parallel
parallel          = 64                  # --max-num-seqs (default 64)
max_input_tokens  = 12288               # in + out = --max-model-len (UNA sequenza)
max_output_tokens = 4096
# kv_cache_type   = "fp8"               # f16 (default) | fp8
# gpu_memory_utilization = 0.90
reasoning_parser  = "qwen3"             # --reasoning-parser
```

Differenze rispetto a llama.cpp:

- vLLM prealloca `gpu_memory_utilization` della VRAM, quindi un modello vLLM
  occupa GPU **intere** e non le condivide. AUTO sceglie il numero di GPU
  (tensor parallel) dai safetensors e dal KV di una sequenza completa. Senza
  NVLink, due modelli da una GPU rendono più di uno su due.
- `parallel` è il numero di sequenze servite in contemporanea e vale anche come
  concorrenza della batch API per quel modello. Il `parallel` di `[defaults]` è
  uno slot di llama.cpp e per vLLM non vale.
- `reasoning_format` e `reasoning_budget` sono solo di llama.cpp. Per i modelli
  con thinking si usa `reasoning_parser`. `enable_thinking` dal client funziona
  uguale, e `repeat_penalty` viene tradotto in `repetition_penalty`.
- Il primo avvio richiede qualche minuto (torch.compile e CUDA graph, poi in
  cache in `~/.cache/vllm`). Cerberus aspetta fino a
  `CERBERUS_HEALTH_TIMEOUT_VLLM` secondi (default 1500).
- Il container parte con `--cleanenv`, con `HF_HUB_OFFLINE=1` e solo le GPU
  assegnate: i pesi devono essere già in cache (`cerberus download`).

Demo con i due backend insieme: `test/demo/unina_h100_vllm/`.
