#!/usr/bin/env bash
#
# new-exp.sh — crea il workspace di un esperimento vLLM offline sopra l'env base.
#
#   source new-exp.sh exp-x [pacchetti...]   crea il workspace e lo attiva
#   bash   new-exp.sh exp-x [pacchetti...]   lo crea e stampa come attivarlo
#
# L'env base si crea una volta con envs/vllm-base.yml (docs/esperimenti_offline.md).
#
# Crea:
#   ~/venvs/exp-x         venv leggero con --system-site-packages (vede torch, vllm, ...)
#   ~/experiments/exp-x   cartella di lavoro: configs/ scripts/ results/ logs/, env.sh, run.sbatch
#
# I pacchetti extra si installano con i vincoli dell'env base (pip.conf nel venv),
# quindi pip non può aggiornare o duplicare torch, vllm e le loro dipendenze.
#
# Variabili (con i default):
#   VLLM_BASE     ~/miniforge3/envs/vllm-base
#   VENVS_DIR     ~/venvs
#   EXPS_DIR      ~/experiments
#   CONSTRAINTS   $VENVS_DIR/vllm-base.constraints.txt   (generato se manca)

_new_exp_main() {
    local name="${1:-}"
    if [ -z "$name" ] || [ "$name" = "-h" ] || [ "$name" = "--help" ]; then
        sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        return 2
    fi
    shift
    if ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "nome non valido: '$name' (lettere, cifre, . _ -)" >&2; return 2
    fi

    local base="${VLLM_BASE:-$HOME/miniforge3/envs/vllm-base}"
    local base_py="${VLLM_BASE_PY:-$base/bin/python}"
    local venvs="${VENVS_DIR:-$HOME/venvs}"
    local exps="${EXPS_DIR:-$HOME/experiments}"
    local constraints="${CONSTRAINTS:-$venvs/vllm-base.constraints.txt}"
    local venv="$venvs/$name"
    local exp="$exps/$name"

    # 1. controlli sull'env base
    [ -x "$base_py" ] || { echo "env base non trovato: $base_py" >&2; return 1; }
    if "$base_py" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)'; then
        echo "$base_py è un venv: la base deve essere un env conda (o un Python vero)," >&2
        echo "altrimenti --system-site-packages vede il Python sottostante, non vllm." >&2
        return 1
    fi
    local versions
    versions="$("$base_py" - <<'PY'
import importlib.metadata as m, sys
out = [f"python={sys.version.split()[0]}"]
for p in ("vllm", "torch", "transformers"):
    try:
        out.append(f"{p}={m.version(p)}")
    except m.PackageNotFoundError:
        out.append(f"{p}=MANCANTE")
print(" ".join(out))
PY
)" || return 1
    case "$versions" in
        *"vllm=MANCANTE"*|*"torch=MANCANTE"*)
            echo "attenzione: nell'env base manca vllm o torch ($versions)" >&2 ;;
    esac
    if [ -e "$venv" ] || [ -e "$exp" ]; then
        echo "esiste già: $venv o $exp (scegli un altro nome o rimuovili)" >&2; return 1
    fi

    # 2. vincoli: le versioni esatte della base, una volta sola per tutti gli esperimenti
    mkdir -p "$venvs" "$exps" || return 1
    if [ ! -s "$constraints" ]; then
        echo "[new-exp] genero i vincoli della base in $constraints"
        "$base_py" -m pip list --format=freeze --exclude-editable 2>/dev/null \
            | grep -viE '^(pip|setuptools|wheel)==' > "$constraints" || return 1
    fi

    # 3. il venv leggero, con il comando concordato
    echo "[new-exp] creo $venv"
    "$base_py" -m venv --system-site-packages "$venv" || return 1
    printf '[install]\nconstraint = %s\n' "$constraints" > "$venv/pip.conf"
    # con --system-site-packages Python riabilita anche ~/.local: lo spegniamo
    cat >> "$venv/bin/activate" <<'ACT'

# aggiunto da new-exp.sh: niente pacchetti da ~/.local dentro gli esperimenti
export PYTHONNOUSERSITE=1
ACT

    # 4. la cartella di lavoro
    mkdir -p "$exp"/{configs,scripts,results,logs} || return 1
    echo "$versions base=$base created=$(date -Iseconds)" > "$exp/.base-versions"
    : > "$exp/requirements.txt"

    cat > "$exp/env.sh" <<ENV
# source ~/experiments/$name/env.sh   (attiva venv e variabili dell'esperimento)
source "$venv/bin/activate"
# dopo il venv, i comandi della base (ninja, nvcc, ...) che vLLM e Triton lanciano
export PATH="$venv/bin:$base/bin:\$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="\${HF_HOME:-\$HOME/.cache/huggingface}"
export VLLM_NO_USAGE_STATS=1
export EXP_DIR="$exp"
ENV

    cat > "$exp/run.sbatch" <<SB
#!/bin/bash
#SBATCH -J $name
#SBATCH -p ingnormal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o $exp/logs/%x-%j.out
source "$exp/env.sh"
cd "$exp"
srun python scripts/smoke.py
SB

    cat > "$exp/scripts/smoke.py" <<'PY'
"""Prova minima: vLLM offline dal venv dell'esperimento."""
import vllm
import torch
from vllm import LLM, SamplingParams

print("vllm", vllm.__version__, "da", vllm.__file__)
print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.device_count())
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.3, max_model_len=4096)
out = llm.generate(["Scrivi una frase su Napoli."], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)
PY

    cat > "$exp/README.md" <<MD
# $name

Creato il $(date +%F) sopra \`$base\`.
Versioni della base: \`$versions\`

- Attiva: \`source $exp/env.sh\`
- Pacchetti extra: aggiungili a \`requirements.txt\` e fai \`pip install -r requirements.txt\`
- Lock dei soli pacchetti dell'esperimento: \`pip freeze --local > requirements.lock\`
- Job: \`sbatch run.sbatch\`
MD

    # 5. pacchetti extra (opzionali), già vincolati dal pip.conf del venv
    if [ "$#" -gt 0 ]; then
        printf '%s\n' "$@" >> "$exp/requirements.txt"
        echo "[new-exp] installo: $*"
        "$venv/bin/python" -m pip install -r "$exp/requirements.txt" || {
            echo "pip install fallito: probabilmente un pacchetto chiede versioni diverse da quelle della base." >&2
            return 1
        }
    fi
    "$venv/bin/python" -m pip freeze --local > "$exp/requirements.lock" 2>/dev/null

    # 6. verifica: torch e vllm devono arrivare dalla base, non dal venv
    PYTHONNOUSERSITE=1 "$venv/bin/python" - "$venv" <<'PY'
import importlib.util, sys
venv = sys.argv[1]
for p in ("torch", "vllm"):
    spec = importlib.util.find_spec(p)
    if spec is None:
        print(f"  {p}: non trovato"); continue
    where = "VENV (shadowing della base!)" if spec.origin.startswith(venv) else "base"
    print(f"  {p}: {where}  {spec.origin}")
PY
    echo "[new-exp] venv: $(du -sh "$venv" | cut -f1)   workspace: $exp"

    # 7. attivazione (solo se lo script è stato lanciato con source)
    if [ "${_NEW_EXP_SOURCED:-0}" = 1 ]; then
        # shellcheck disable=SC1091
        source "$exp/env.sh" && cd "$exp" && echo "[new-exp] attivato: $name"
    else
        echo "[new-exp] per attivarlo:  source $exp/env.sh && cd $exp"
    fi
}

(return 0 2>/dev/null) && _NEW_EXP_SOURCED=1 || _NEW_EXP_SOURCED=0
if [ "$_NEW_EXP_SOURCED" = 1 ]; then
    _new_exp_main "$@"
    _rc=$?; unset -f _new_exp_main; unset _NEW_EXP_SOURCED; return $_rc
else
    _new_exp_main "$@"
    exit $?
fi
