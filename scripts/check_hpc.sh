#!/usr/bin/env bash
#
# check_hpc.sh — raccoglie le informazioni sul nuovo HPC che servono per Cerberus.
#
# Uso:
#   bash check_hpc.sh            # verifiche rapide (qualche minuto al massimo)
#   bash check_hpc.sh --vllm     # in più scarica l'immagine vLLM (~10 GB) e la prova
#
# L'output va a schermo e in hpc_check_<host>_<data>.txt nella directory corrente.
# Non modifica niente fuori da $WORK (default: $HOME/cerberus-check).
#
set -u
USER="${USER:-$(id -un)}"

WITH_VLLM=0
[ "${1:-}" = "--vllm" ] && WITH_VLLM=1

if [ "$WITH_VLLM" = 1 ] && ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    echo "ERRORE: --vllm richiede Apptainer, che qui ($(hostname -s)) non c'è."
    echo "Entra in un nodo GPU, ad esempio:"
    echo "  srun -p ingnormal -N1 --gpus-per-node=2 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash"
    exit 1
fi

WORK="${WORK:-$HOME/cerberus-check}"
mkdir -p "$WORK"
OUT="$(pwd)/hpc_check_$(hostname -s)_$(date +%Y%m%d_%H%M).txt"
exec > >(tee "$OUT") 2>&1

section() { printf '\n==================== %s ====================\n' "$*"; }
run() {  # run <timeout_s> <comando...>: stampa il comando e lo esegue con timeout
    local t="$1"; shift
    printf '\n$ %s\n' "$*"
    timeout "$t" bash -c "$*"
    local rc=$?
    [ $rc -eq 124 ] && echo "[TIMEOUT dopo ${t}s]"
    [ $rc -ne 0 ] && [ $rc -ne 124 ] && echo "[exit $rc]"
    return 0
}

section "Sistema"
run 10 'hostname; whoami; date; uname -r'
run 10 'cat /etc/os-release | head -3'
run 10 'nproc; free -g; lscpu | grep -E "Model name|Socket|NUMA node\(s\)"'
run 10 'ldd --version | head -1'

section "GPU"
run 20 'nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv'
run 20 'nvidia-smi topo -m'
run 20 'nvidia-smi nvlink --status 2>/dev/null | head -40'

section "Scheduler"
run 10 'which sbatch srun salloc squeue scontrol sacctmgr qsub pbsnodes 2>/dev/null; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<nessuno>}"'
if command -v sinfo >/dev/null 2>&1; then
    run 20 'sinfo -s'
    run 20 'sinfo -o "%P %a %l %D %G %c %m %N"'
    run 20 "scontrol show node $(hostname -s) | grep -E 'NodeName|Gres|CPUTot|RealMemory|Partitions|State'"
    run 20 "sacctmgr -n -P show assoc user=$USER format=account,partition,qos"
    run 20 "squeue -u $USER"
    run 20 'scontrol show config | grep -iE "^(SelectType|GresTypes|PrologFlags|MaxJobCount|ClusterName)"'
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        # già dentro un job: un srun qui aprirebbe un passo di questo job, non un job nuovo
        run 10 'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-<unset>}"'
        run 20 "scontrol show job $SLURM_JOB_ID | grep -E 'NodeList|TRES|MinMemory'"
    else
        echo
        echo "# Prova di un job con 1 GPU (si ferma da sola dopo 60 s se resta in coda)"
        run 60 'srun -p ingnormal --gpus=1 --mem=4G --time=00:02:00 nvidia-smi -L'
    fi
else
    echo "sinfo non trovato: niente SLURM su questo nodo (o non nel PATH)."
fi

section "Container (Apptainer/Singularity)"
run 10 'which apptainer singularity enroot podman docker 2>/dev/null'
run 10 'apptainer --version 2>/dev/null || singularity --version'
run 10 "grep \"^$USER:\" /etc/subuid /etc/subgid 2>/dev/null || echo 'nessuna voce subuid/subgid per $USER'"
run 10 'env | grep -iE "^(APPTAINER|SINGULARITY)_" || echo "nessuna variabile APPTAINER_/SINGULARITY_"'
if command -v srun >/dev/null 2>&1; then
    run 10 'srun --help 2>/dev/null | grep -iE "container" || echo "srun senza opzioni --container (niente pyxis)"'
fi
SINGBIN="$(command -v apptainer || command -v singularity || true)"
if [ -n "$SINGBIN" ]; then
    # APPTAINER_TMPDIR già impostato vince: su un disco locale (/tmp del nodo)
    # le conversioni sono molto più veloci che su Lustre
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$WORK/apptainer-cache}"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$WORK/apptainer-tmp}"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    echo
    echo "# Pull da Docker Hub + GPU dentro il container (--nv)"
    run 300 "$SINGBIN exec --nv docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"
    echo
    echo "# Build con --fakeroot (serve per costruire l'immagine llama.cpp)"
    run 300 "$SINGBIN build --force --fakeroot '$WORK/fakeroot_test.sif' docker://ubuntu:22.04 && rm -f '$WORK/fakeroot_test.sif' && echo FAKEROOT_OK"
fi

section "Storage"
run 20 "df -h \$HOME /scratch* /work* /lustre* /data* /archive* /nfs* /gpfs* /beegfs* 2>/dev/null"
run 20 'findmnt -t nfs,nfs4,lustre,gpfs,beegfs,cephfs,xfs,ext4 -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL 2>/dev/null'
run 20 'quota -s 2>/dev/null || echo "quota non disponibile"'
run 20 "command -v lfs >/dev/null && lfs quota -h -u $USER \$HOME 2>/dev/null || true"
run 10 'echo "HOME=$HOME TMPDIR=${TMPDIR:-<unset>} HF_HOME=${HF_HOME:-<unset>}"; df -h /tmp /dev/shm'

section "Moduli e software"
if type module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; then
    [ -f "${MODULESHOME:-/usr/share/Modules}/init/bash" ] && . "${MODULESHOME:-/usr/share/Modules}/init/bash"
    run 30 'module avail 2>&1 | grep -iE "cuda|python|conda|anaconda|miniforge|gcc|cmake|apptainer|singularity|nccl" | head -60'
else
    echo "comando 'module' non disponibile"
fi
run 10 'which nvcc gcc cmake python3 conda mamba micromamba pip3 git curl wget 2>/dev/null'
run 10 'python3 --version; gcc --version 2>/dev/null | head -1; nvcc --version 2>/dev/null | tail -1'

section "Rete"
run 10 'env | grep -iE "_proxy=" || echo "nessun proxy configurato"'
for u in https://huggingface.co https://cdn-lfs.huggingface.co https://pypi.org/simple/ \
         https://registry-1.docker.io/v2/ https://github.com https://developer.download.nvidia.com; do
    printf '%-45s ' "$u"
    curl -s -o /dev/null -m 10 -w '%{http_code}\n' "$u" || true  # 000 = non raggiungibile
done

section "Porte (per il client Cerberus)"
run 10 "ss -ltn 2>/dev/null | head -20"
run 10 'cat /proc/sys/net/ipv4/ip_local_port_range'
echo "Per provare l'accesso da un'altra macchina: python3 -m http.server 8081 qui,"
echo "poi da lì: curl -s -o /dev/null -w '%{http_code}\n' http://$(hostname -s):8081"

if [ "$WITH_VLLM" = 1 ] && [ -n "$SINGBIN" ]; then
    section "vLLM (container ufficiale)"
    export HF_HOME="${HF_HOME:-$WORK/hf-cache}"
    # v0.32.0 = CUDA 13.0, ok col driver 590; alternativa: VLLM_TAG=v0.32.0-cu129
    VSIF="$WORK/vllm_${VLLM_TAG:-v0.32.0}.sif"
    [ -f "$VSIF" ] || run 3600 "$SINGBIN pull '$VSIF' docker://vllm/vllm-openai:${VLLM_TAG:-v0.32.0}"
    if [ -f "$VSIF" ]; then
        run 60 "$SINGBIN exec --nv '$VSIF' python3 -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count())'"
        MODEL=Qwen/Qwen2.5-0.5B-Instruct
        for TP in 1 2; do
            echo
            # TP=2 prova solo che NCCL funzioni tra le GPU: senza NVLink in produzione
            # conviene un'istanza per GPU
            echo "# vllm serve $MODEL con tensor-parallel-size=$TP"
            LOG="$WORK/vllm_tp$TP.log"
            "$SINGBIN" exec --nv --env HF_HOME="$HF_HOME" "$VSIF" \
                vllm serve "$MODEL" --port 8765 --tensor-parallel-size "$TP" \
                --gpu-memory-utilization 0.3 --max-model-len 4096 > "$LOG" 2>&1 &
            PID=$!
            ok=0
            for _ in $(seq 1 120); do
                sleep 5
                kill -0 $PID 2>/dev/null || break
                if curl -s -m 3 localhost:8765/health -o /dev/null -w '%{http_code}' | grep -q 200; then ok=1; break; fi
            done
            if [ $ok = 1 ]; then
                curl -s -m 60 localhost:8765/v1/chat/completions -H 'Content-Type: application/json' \
                  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Rispondi solo: ciao\"}],\"max_tokens\":10}"
                echo; echo "VLLM_TP${TP}_OK"
            else
                echo "VLLM_TP${TP}_FAIL — ultime righe del log:"
                tail -40 "$LOG"
            fi
            kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
        done
    fi
fi

section "Fine"
echo "Output salvato in: $OUT"
