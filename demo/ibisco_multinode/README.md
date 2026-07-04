# Cerberus multinode — demo su IBiSCo (2 nodi, 4 modelli)

Estensione **multi-nodo** della [demo IBiSCo](../ibisco/README.md): un unico job
SLURM alloca **2 nodi di calcolo con 3 GPU ciascuno** e avvia **4 modelli LLM**,
poi li interroga in sequenza dal nodo master salvando le risposte.

## Layout

Ogni nodo ospita due modelli — uno su una singola GPU e uno **splittato su due
GPU** — usando così tutte e 3 le GPU del nodo:

| Modello     | Nodo            | GPU (logiche del nodo) | Porta |
|-------------|-----------------|------------------------|-------|
| `n0-single` | 0 (master)      | `CUDA0`                | 8081  |
| `n0-split`  | 0 (master)      | `CUDA1,CUDA2`          | 8082  |
| `n1-single` | 1 (worker)      | `CUDA0`                | 8083  |
| `n1-split`  | 1 (worker)      | `CUDA1,CUDA2`          | 8084  |

Il **nodo master** è il primo di `$SLURM_JOB_NODELIST` (dove parte lo script):
esegue i suoi 2 modelli localmente, avvia i 2 modelli del **worker** da remoto e
interroga tutti e 4.

## Come funziona

- **Un job, più nodi.** `run_demo_multinode.sh` gira sul master dentro un job
  SLURM con ≥ 2 nodi. Ricava i nomi dei nodi con `scontrol show hostnames`.
- **Manifest condiviso.** Il master risolve i `.gguf` nella cache HF e genera
  `.runtime.tsv` (quale modello, su quale nodo/GPU, quale porta) e `.endpoints.tsv`
  (dove raggiungere ogni modello). La cartella è su Lustre, quindi visibile a
  tutti i nodi.
- **Avvio per nodo via `srun`.** Per ogni nodo viene lanciato uno step
  `srun --nodes=1 --nodelist=<nodo> --gres=gpu:3 --overlap node_launch.sh <idx>`
  in background. `node_launch.sh` avvia i server di quel nodo (`singularity exec`
  sulla sandbox, binari su Lustre, `--host 0.0.0.0`) e resta in `wait`, mantenendo
  vivo lo step finché i server sono attivi.
- **Raggiungibilità tra nodi.** I server ascoltano su `0.0.0.0`; dal master si
  raggiungono a `http://<hostname_nodo>:<porta>` sulla rete del cluster.
- **Interrogazione.** `query_models.py` legge `.endpoints.tsv` e interroga i 4
  modelli in sequenza (thinking e risposta finale separati, come nelle altre demo).
- **Arresto pulito.** All'uscita il master uccide gli step `srun`, che a loro
  volta chiudono i server su ciascun nodo.

## Prerequisiti

- Sandbox CUDA e binari di llama.cpp su Lustre (vedi
  [../../guides/llamacpp_install.md](../../guides/llamacpp_install.md), Parte 2);
  default `~/tools/cuda-build` e `~/tools/llama.cpp-master/build/bin`.
- Modelli GGUF in cache HF (condivisa tra i nodi). Scaricabili con
  [../ibisco/download_models.sh](../ibisco/download_models.sh).

## Esecuzione

Batch:

```bash
cd demo/ibisco_multinode
sbatch submit.sbatch      # 2 nodi, 3 GPU/nodo (adattare partizione/tempo)
```

Interattivo:

```bash
salloc --nodes=2 --ntasks-per-node=1 --gres=gpu:3 --time=01:00:00
cd demo/ibisco_multinode
./run_demo_multinode.sh
```

Le risposte finiscono in `outputs/responses_<timestamp>.{json,md}`; i log dei
server in `logs/<nome>.log`, quelli degli step srun in `logs/node_<idx>_srun.log`.

### Parametri sovrascrivibili

Stessi della demo single-node (`CERBERUS_SANDBOX`, `CERBERUS_BIN`, `CERBERUS_BIND`,
`HF_HOME`, `NGL`, `THREADS`, `THREADS_HTTP`, `HEALTH_TIMEOUT`) più:

- `GPUS_PER_NODE` — GPU richieste allo step srun per nodo (default: 3).

## Personalizzazione

- **Modelli / layout GPU**: `models.conf` (campi `node` e `cuda_device`). Per
  aggiungere un terzo nodo basta allocarlo nel job e usare `node=2` nelle righe.
- **Modelli grandi con split reale**: le righe `*-split` qui usano Qwen3-4B a
  scopo illustrativo (entra in una sola GPU); puntandole a un modello che eccede i
  32 GB di una V100, lo split su due GPU diventa quello vero — il meccanismo è
  identico.

## Note e limiti

- I flag di `srun` (`--gres`, `--overlap`, `--nodelist`) seguono lo schema SLURM
  standard ma potrebbero richiedere un adattamento alla configurazione specifica
  di IBiSCo (nome della partizione, policy sugli step concorrenti).
- La risoluzione degli hostname dei nodi e la raggiungibilità delle porte tra
  nodi dipendono dalla rete del cluster: sono attese funzionanti in un job SLURM,
  ma vanno confermate al primo run reale.
