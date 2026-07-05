# Cerberus demo — multi node (IBiSCo)

Serves **4 models across 2 nodes** using the Cerberus tool, with **automatic
node + GPU placement**, then queries them through the OpenAI-compatible client.
See the full tool guide in [../../guides/cerberus_tool.md](../../guides/cerberus_tool.md).

## What makes it multi-node
`gpus_per_node = 3`. Two models are `MANUAL num_gpus = 2` (tensor-split over two
whole GPUs), so together they need 4 GPUs > 3 → Cerberus spreads them onto **two
nodes**; the two small `AUTO` models pack into leftover GPU space. You don't pin
anything — the tool decides placement. (A 4B model split over 2 GPUs is only
illustrative; the same path serves models bigger than a single 32 GB V100.)

## Files
| File | Purpose |
|------|---------|
| `models.conf` | 4 models, `gpus_per_node = 3` (forces 2 nodes) |
| `submit.sbatch` | SLURM batch job (2 nodes × 3 GPUs) → runs `run_demo.sh` |
| `run_demo.sh` | download → `cerberus up` → client → teardown |
| `demo_client.py` | queries every model via `client_llamacpp` |
| `prompts.txt` | prompts |

## Run
Batch:
```bash
# set HF_TOKEN and activate the cerberus env inside submit.sbatch first
sbatch submit.sbatch
tail -f logs/slurm-<jobid>.out
```
Interactive:
```bash
conda activate cerberus
salloc --partition=gpus --nodes=2 --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=01:00:00
./run_demo.sh
```

Cerberus resolves the 2 allocated nodes, packs the 4 models, launches each node's
servers via `srun` (each on a free port, `--host 0.0.0.0`), writes `endpoints.json`
here, and the client queries all 4 from the master. Answers in
`outputs/responses_<ts>.{json,md}`; per-server logs in `.cerberus/<jobid>/logs/`.
