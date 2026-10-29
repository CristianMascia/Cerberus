# Cerberus demo — HPC UniNA (1 nodo, 2x H100)

Serve **Llama-3.3-70B (Q4_K_M) e un modello piccolo** su un nodo della partizione
`ingnormal`, poi li interroga col client. Sono gli stessi modelli della demo
IBiSCo: lì il 70B va diviso su 2 V100, qui sta su **una sola H100**, con 4 slot
da 16k token. Guida completa: [../../../docs/hpc_unina_h100.md](../../../docs/hpc_unina_h100.md).

## Prerequisiti
- env `cerberus` (Miniforge, `pip install -e .`), `HF_TOKEN` impostato;
- immagine `~/containers/llamacpp_h100.sif` (sm_90), vedi la guida.

## Esecuzione
```bash
conda activate cerberus
salloc --partition=ingnormal --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 \
       --cpus-per-task=10 --mem=200G --time=01:00:00
./run_demo.sh
```

`run_demo.sh` carica `profiles/unina-h100.env`, scarica i GGUF (~40 GiB la prima
volta), avvia i server con `cerberus up`, lancia `../demo_client.py` e spegne
tutto. Le risposte finiscono in `outputs/`.
