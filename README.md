# Cerberus

Serving dichiarativo di modelli LLM su cluster SLURM (IBiSCo, sibisco). In un
solo `models.conf` (TOML) dichiari quali modelli servire e quanto contesto
serve. Cerberus calcola le GPU necessarie, posiziona i modelli su nodi e GPU
dell'allocazione e avvia i server: `llama-server` per i GGUF, `vllm serve` per i
safetensors. Infine scrive `endpoints.json`, che il client usa per instradare le
richieste per **label**.

## Uso

```bash
cerberus validate -c models.conf     # schema + stima GPU/nodi
cerberus download -c models.conf     # pesi nella cache HF (dal login node)
salloc ...                           # l'allocazione la crei tu
cerberus up -c models.conf           # avvia i server, scrive endpoints.json, resta in servizio
cerberus status                      # UP/DOWN di ogni server
cerberus batch submit|run|status|merge ...   # batch API, vedi docs/batch.md
```

```python
from cerberus import CerberusClient
c = CerberusClient()                                   # legge ./endpoints.json
r = c.chat("qwen3-8b", [{"role": "user", "content": "Ciao"}], reasoning=False)
print(r.content, r.reasoning)
rs = c.chat_batch("qwen3-8b", [msgs1, msgs2, ...])     # richieste in parallelo
```

`models.conf.example` è l'esempio annotato con tutti i campi. I principali sono:

- `backend = "llamacpp"` (GGUF, `gguf_file`) oppure `"vllm"` (safetensors di
  `hf_repo`). Un modello vLLM occupa GPU intere.
- `alloc_mode = "AUTO"` (GPU calcolate dai pesi e dal contesto) oppure `"MANUAL"`
  con `num_gpus`.
- `max_input_tokens` + `max_output_tokens` è il contesto di una richiesta;
  `parallel` è il numero di richieste servite insieme.
- `replicas = N` crea N server identici, ognuno sulla sua GPU; il client alterna
  le richieste tra le copie.

Su sibisco l'ambiente è in `profiles/unina-h100.env` (immagini `.sif`,
`HF_HOME`, token). Il tensor parallel di vLLM su 2 GPU lì non funziona (P2P tra
le GPU rotto), mentre le repliche sì.

## Altri strumenti

- `scripts/bench.py`: benchmark di un modello servito (`run`) e confronto tra
  risultati (`compare`); configurazioni e job in `test/bench/`.
- `scripts/new-exp.sh`: crea un esperimento vLLM offline (venv sopra l'env
  `vllm-base`, cartella in `~/experiments/`).

## Struttura

| Percorso | Contenuto |
| --- | --- |
| `cerberus/` | config, stima VRAM (`estimator`, `vram`, `vllm_backend`), `placement`, `download`, `deploy` + `node_launch.sh`, `client`, `batch`, `cli` |
| `profiles/` | variabili d'ambiente per cluster |
| `scripts/` | `bench.py`, `new-exp.sh` |
| `test/unit/` | test unitari (niente rete, GPU o SLURM) |
| `test/demo/` | demo d'integrazione: IBiSCo 1 e 2 nodi, sibisco llama.cpp e vLLM, batch in locale |
| `test/bench/` | configurazioni e job del benchmark |
| `docs/batch.md` | formato dell'input e comandi della batch API |

## Test

```bash
pip install -e .
python -m unittest discover -s test/unit
```
