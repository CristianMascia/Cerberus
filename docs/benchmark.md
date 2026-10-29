# Benchmark: IBiSCo contro HPC UniNA, llama.cpp contro vLLM

Tre misure con lo stesso modello (Qwen3-8B) e lo stesso carico:

| Tag | Dove | Backend | Pesi | Cosa misura |
| --- | --- | --- | --- | --- |
| `ibisco-llamacpp` | IBiSCo, 1 V100 32 GB | llama.cpp | GGUF Q8_0 | riferimento (il setup vecchio) |
| `unina-llamacpp` | UniNA, 1 H100 NVL | llama.cpp | GGUF Q8_0 | speedup dovuto all'hardware |
| `unina-vllm` | UniNA, 1 H100 NVL | vLLM | safetensors bf16 | differenza tra i backend |

Ogni misura usa una sola GPU. Il carico di `scripts/bench.py` è sempre lo
stesso: prompt sintetici con seme fisso (circa 1024 token in ingresso; il numero
vero è in `mean_in_tokens`), 256 token in uscita forzati con `ignore_eos`,
temperatura 0, thinking spento, richieste in streaming. Per ogni concorrenza
(default 1, 4, 16, 32, 64) manda `max(16, 4 x concorrenza)` richieste dopo due
di riscaldamento.

## Metriche

- **tok/s**: token generati al secondo da tutto il server (throughput).
- **req/s**: richieste completate al secondo.
- **TTFT**: tempo al primo token (attesa in coda + elaborazione del prompt).
- **TPOT**: tempo per ogni token dopo il primo (velocità di generazione vista
  da una richiesta).
- **e2e**: latenza totale di una richiesta. Di tutte: p50 e p95.

## UniNA (i due backend in un'unica allocazione)

```bash
conda activate cerberus
cd ~/Projects/Cerberus/test/bench
cerberus validate -c unina.conf && cerberus download -c unina.conf    # login
salloc -p ingnormal -N1 --ntasks-per-node=1 --gpus-per-node=2 \
       --cpus-per-task=10 --mem=200G --time=02:00:00
cerberus up -c unina.conf &          # aspetta "ready" (vLLM: qualche minuto al primo avvio)
BENCH_GPU=H100 python ../../scripts/bench.py run --model qwen3-8b-llamacpp --tag unina-llamacpp
BENCH_GPU=H100 python ../../scripts/bench.py run --model qwen3-8b-vllm --tag unina-vllm \
       --concurrency 1,4,16,32,64,128
kill %1
```

## IBiSCo

Stessa repo (`git pull` su main), stesso GGUF e stessi 32 slot:

```bash
cd ~/Cerberus/test/bench             # dove sta la repo su IBiSCo
cerberus download -c ibisco.conf
salloc ... (come per le altre demo IBiSCo, 1 GPU basta)
cerberus up -c ibisco.conf &
BENCH_GPU=V100 python ../../scripts/bench.py run --model qwen3-8b-llamacpp --tag ibisco-llamacpp
kill %1
```

## Confronto

Copia i `bench_results.jsonl` in un'unica cartella e:

```bash
python scripts/bench.py compare ibisco/bench_results.jsonl unina/bench_results.jsonl \
       --baseline ibisco-llamacpp
```

Stampa la tabella completa e lo speedup di throughput di ogni prova rispetto a
IBiSCo, a parità di concorrenza.

## Come leggere i numeri

- A concorrenza 1 conta la banda di memoria: si vede soprattutto lo speedup
  dell'hardware (H100 contro V100) nel TPOT.
- A concorrenza alta conta il batching: qui vLLM dovrebbe staccare llama.cpp.
  Oltre 32 richieste llama.cpp le mette in coda (32 slot), e si vede dal TTFT.
- Q8_0 pesa circa metà del bf16: a concorrenza bassa questo favorisce
  llama.cpp. Per un confronto alla pari dei pesi, si può rifare la prova vLLM
  con un checkpoint FP8.
- Il primo avvio di vLLM compila (torch.compile, CUDA graph): il benchmark
  parte solo dopo "ready" e fa due richieste di riscaldamento, quindi questo
  tempo non entra nelle misure.
