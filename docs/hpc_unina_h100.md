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
| CPU / RAM per nodo | | 256 / 773 GB |
| Container | Singularity, sandbox + binari in `~/tools` | Apptainer 1.4, `.sif` con llama-server dentro |
| Home | Lustre | Lustre (`/lustre/home/$USER`, senza quota) |
| Python | conda | Python 3.9 di sistema, niente moduli: serve Miniforge |

L'accounting di SLURM è spento, quindi non serve `--account`. Non c'è limite di
tempo sulle partizioni.

## 1. Python (una volta, sul login node)

```bash
curl -L -o Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p $HOME/miniforge3 && rm Miniforge3.sh
source $HOME/miniforge3/bin/activate
cd ~/Cerberus
conda env create -f environment.yml && conda activate cerberus
pip install -e .
```

## 2. Immagine llama.cpp per H100 (una volta, su un nodo GPU)

La build con `--fakeroot` funziona anche senza voce in `/etc/subuid`. Va fatta
su un nodo GPU così `nvidia-smi` rileva `9.0`; `CUDA_ARCH=90` la rende esplicita.

```bash
salloc -p ingnormal -N1 --gpus-per-node=1 --cpus-per-task=32 --mem=64G --time=01:00:00
cd ~/Cerberus/cerberus/containers
CUDA_ARCH=90 ./build_llamacpp_local.sh llamacpp_local.def llamacpp_h100.sif
```

I warning `xattr ... ENOTSUP` durante la build sono dovuti a Lustre e sono
innocui. Il driver 590 (CUDA 13.1) esegue senza problemi l'immagine basata su
CUDA 12.4.

## 3. Configurazione

```bash
source ~/Cerberus/profiles/unina-h100.env   # CERBERUS_IMAGE, HF_HOME, thread
```

Nel `models.conf`:

```toml
[allocation]
gpus_per_node = 2
vram_gib      = 92      # GiB utilizzabili di UNA GPU (default 32 = V100)

[batch]
partition     = "ingnormal"
cpus_per_task = 32
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
       --cpus-per-task=32 --mem=200G --time=04:00:00
source ~/Cerberus/profiles/unina-h100.env
cerberus up
```

La batch API (`cerberus batch submit`) usa `partition` e `cpus_per_task` della
sezione `[batch]`. La demo completa è in `test/demo/unina_h100/`.
