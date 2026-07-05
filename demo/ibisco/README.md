# Cerberus demo — single node (IBiSCo)

Serves **a Llama-3.3-70B model (auto-split over 2 GPUs) plus a small model, on one
node** using the Cerberus tool, then queries them through the OpenAI-compatible
client. See the full tool guide in
[../../guides/cerberus_tool.md](../../guides/cerberus_tool.md).

The 70B Q4_K_M (~40 GiB) does not fit on a single 32 GB V100, so AUTO gives it
**2 GPUs** (tensor-split); the small `qwen-coder` packs onto the third GPU — all on
one node. (First run downloads ~40 GiB.)

## Files
| File | Purpose |
|------|---------|
| `models.conf` | declarative spec: 70B + a small model, `gpus_per_node = 3` |
| `run_demo.sh` | download → `cerberus up` → client → teardown |
| `demo_client.py` | queries every model via `client_llamacpp` (thinking separated) |
| `prompts.txt` | prompts sent to each model |

## Prerequisites
- `cerberus` conda env installed (`pip install -e .` from the repo root) with `HF_TOKEN` set.
- CUDA sandbox + llama.cpp binaries under `~/tools` (see the install guide).

## Run
```bash
conda activate cerberus
salloc --partition=gpus --nodes=1 --ntasks-per-node=1 --gpus-per-node=3 \
       --cpus-per-task=8 --time=01:00:00
./run_demo.sh
```

`run_demo.sh` downloads the GGUFs, deploys the servers (Cerberus auto-places the
three models onto the node's GPUs — small ones may share a GPU), writes
`endpoints.json` here, runs `demo_client.py`, and tears everything down. Answers
land in `outputs/responses_<ts>.{json,md}`.

## Manual, step by step
```bash
cerberus validate            # GPU/node estimate
cerberus download            # fetch GGUFs
cerberus up &                # deploy; writes endpoints.json, keeps serving
python demo_client.py        # or use client_llamacpp from your own code
cerberus status              # UP/DOWN
# Ctrl-C the 'cerberus up' to stop
```
