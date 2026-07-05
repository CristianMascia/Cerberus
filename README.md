# Cerberus

Declarative multi-model **llama.cpp** serving on the **IBiSCo** HPC cluster
(SLURM, Singularity, V100-SXM2-32GB, Lustre).

You write one TOML file — `models.conf` — declaring the models to serve. Cerberus
handles the rest: downloading the GGUFs, deciding how many GPUs each model needs,
packing them onto nodes/GPUs, launching the `llama-server` instances, and writing
an endpoint map that a Python client uses to route requests **by label** over the
OpenAI-compatible API.

## Quick start
```bash
conda create -n cerberus python=3.11 -y && conda activate cerberus
pip install -e .
export HF_TOKEN=hf_...

cerberus validate          # schema + GPU/node estimate
cerberus download          # fetch the GGUFs into the HF cache
salloc --partition=gpus --nodes=<N> --ntasks-per-node=1 --gpus-per-node=3 ...
cerberus up                # deploy; writes endpoints.json, keeps serving
```
```python
from client_llamacpp import CerberusClient
c = CerberusClient()
print(c.chat("qwen-big", [{"role":"user","content":"Ciao"}]).content)
```

## Layout
| Path | What |
|------|------|
| `cerberus/` | the tool: `config`, `vram`, `placement`, `download`, `deploy`, `cli`, `node_launch.sh` |
| `client_llamacpp.py` | OpenAI-compatible client (routes by label, separates reasoning) |
| `vram_estimator/` | GGUF header parser + KV/VRAM estimator (used by placement) |
| `models.conf.example` | annotated config example |
| `guides/` | guide dettagliate: [panoramica](guides/cerberus_panoramica.md), [demo](guides/cerberus_esecuzione_demo.md), [progetto](guides/cerberus_progetto.md); + install/download/usage di llama.cpp |
| `containers/` | Singularity image / sandbox build recipes |
| `demo/` | worked examples: `ibisco` (single node), `ibisco_multinode` (2 nodes) |

See [guides/cerberus_tool.md](guides/cerberus_tool.md) for the full workflow.
