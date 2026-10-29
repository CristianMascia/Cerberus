# Cerberus demo — HPC UniNA, vLLM e llama.cpp insieme (1 nodo, 2x H100)

Un solo `cerberus up` con due backend: **Qwen3-8B servito da vLLM** su una H100
intera e un modello GGUF piccolo su **llama.cpp** nell'altra. Il client li
interroga allo stesso modo.

## Prerequisiti
- env `cerberus` (Miniforge, `pip install -e .`);
- `~/containers/llamacpp_h100.sif` e `~/containers/vllm-0.32.0.sif`.

## Esecuzione
```bash
conda activate cerberus
salloc --partition=ingnormal --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 \
       --cpus-per-task=10 --mem=200G --time=01:00:00
./run_demo.sh
```

Il primo avvio di vLLM richiede qualche minuto in più (torch.compile e cattura
dei CUDA graph, poi in cache in `~/.cache/vllm`). Log dei server in
`.cerberus/<job>/logs/`, risposte in `outputs/`.
