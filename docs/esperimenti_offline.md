# Esperimenti offline con vLLM (sibisco)

Cerberus serve i modelli come server (`cerberus up`, batch API). Per gli
esperimenti su un solo modello, con la classe `LLM` di vLLM dentro il proprio
script Python, si lavora **fuori da Cerberus**. Si usa la stessa cache HF
(`~/.cache/huggingface`) e la stessa versione di vLLM dei server (0.32.0).

Lo schema è a due livelli:

- **`vllm-base`**: un env conda con vLLM, torch e dipendenze. Si crea una volta
  sola e poi non si tocca più.
- **un venv per esperimento**, sopra la base, con `--system-site-packages`. Vede
  torch e vllm della base senza copiarli (pochi MB invece di ~10 GB) e contiene
  solo i pacchetti in più dell'esperimento. I pacchetti si installano con i
  vincoli della base, quindi pip non può aggiornare o duplicare torch e vllm.

## 1. Env base (una volta)

L'installazione scarica circa 10 GB di wheel e va fatta **dal login node**. Sui
nodi di calcolo pip fallisce scrivendo su Lustre (`OSError: [Errno 14] Bad
address`), mentre dal login funziona. Ci vogliono circa 5 minuti:

```bash
cd ~/Projects/Cerberus
conda env create -f envs/vllm-base.yml          # solo Python 3.12
export TMPDIR=/tmp/$USER-pip; mkdir -p $TMPDIR  # ~6 GB al picco, poi si cancella
~/miniforge3/envs/vllm-base/bin/python -m pip install --no-cache-dir \
    -r envs/vllm-base-requirements.txt 2>&1 | tee ~/vllm-base-pip.log
rm -rf /tmp/$USER-pip; unset TMPDIR
```

Poi, come job sulla GPU, la verifica, il primo esperimento `exp-prova` e lo smoke
test (non serve restare collegati):

```bash
~/miniforge3/envs/vllm-base/bin/hf download Qwen/Qwen2.5-0.5B-Instruct   # pesi dello smoke test, dal login
cd ~/Projects/Cerberus && sbatch envs/setup_vllm_base.sbatch
tail -f vllm-base-setup-<job>.out    # alla fine: "0.32.0 2.13... cuda True", una frase, "[setup] fatto"
```

Da qui in poi nella base non si installa più niente: per un pacchetto nuovo si crea un esperimento.

## 2. Nuovo esperimento

```bash
source ~/Projects/Cerberus/scripts/new-exp.sh exp-x            # crea e attiva
source ~/Projects/Cerberus/scripts/new-exp.sh exp-y peft datasets   # con pacchetti extra
```

Lo script fa questo:

- crea `~/venvs/exp-x`, con il comando
  `~/miniforge3/envs/vllm-base/bin/python -m venv --system-site-packages ~/venvs/exp-x`;
- scrive i vincoli della base in `~/venvs/vllm-base.constraints.txt` (alla prima
  esecuzione) e li attiva nel `pip.conf` del venv;
- crea `~/experiments/exp-x` con `configs/`, `scripts/`, `results/`, `logs/`,
  `env.sh`, `run.sbatch`, `requirements.txt`, `requirements.lock` e uno
  `scripts/smoke.py`;
- controlla che torch e vllm arrivino dalla base, e non da una copia nel venv.

Per riprendere un esperimento in seguito:

```bash
source ~/experiments/exp-x/env.sh && cd ~/experiments/exp-x
```

## 3. Lancio

Il codice dell'esperimento va in `scripts/`. Prova prima la catena con lo smoke
test, che usa Qwen2.5-0.5B su una GPU:

```bash
cd ~/experiments/exp-x
sbatch run.sbatch                 # log in logs/exp-x-<job>.out
```

`run.sbatch` chiede 1 GPU e 8 CPU su `ingnormal`; modifica l'ultima riga per
lanciare il tuo script. Per lavorare in interattivo:
`salloc ... --gpus-per-node=1`, poi `srun --pty bash -l` e `source env.sh`.

## Regole pratiche

- **Installazioni e download dal login node.** `new-exp.sh` con pacchetti extra,
  `pip install` nei venv e i download dei pesi vanno fatti dal login: sui nodi di
  calcolo le scritture grosse su Lustre possono fallire con `Errno 14 Bad address`.
- **Mai `pip install` nella base.** Un pacchetto che vuole un'altra versione di
  torch o di transformers fa fallire l'installazione nel venv: è voluto. In quel
  caso serve una seconda base (es. `vllm-base-0.33`) con il suo yml.
- **Riproducibilità:** `requirements.lock` (`pip freeze --local`) contiene solo
  i pacchetti dell'esperimento, mentre `.base-versions` registra la base usata.
  Insieme ricostruiscono l'ambiente. Salvali con i risultati.
- **Una GPU per esperimento.** Il tensor parallel su 2 GPU su questi nodi si
  blocca (vedi [hpc_unina_h100.md](hpc_unina_h100.md#repliche-e-tensor-parallel)):
  usa modelli che entrano in una H100, eventualmente in FP8.
- **Pesi:** si usa la cache comune `~/.cache/huggingface`. Ciò che Cerberus ha già
  scaricato è riusato, e per i modelli nuovi fai prima
  `hf download <repo>` dal login node.
- **Cache di vLLM:** torch.compile e CUDA graph finiscono in `~/.cache/vllm`. Il
  primo avvio di un modello è lento, i successivi no.
- **Niente `~/.local`:** `env.sh` imposta `PYTHONNOUSERSITE=1`, così pacchetti
  installati con `pip install --user` non entrano negli esperimenti.
- **Pulizia:** un esperimento finito si cancella con
  `rm -rf ~/venvs/exp-x ~/experiments/exp-x` (prima salva `results/`).
