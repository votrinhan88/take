#!/bin/bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/.env" || true

EXPERIMENT_NAMES=(embed classify classify2 condense finetune generate)
declare -A EXPERIMENTS=(
    [embed]="expts/embed.py"
    [classify]="expts/classify.py"
    [classify2]="expts/classify2.py"
    [condense]="expts/condense.py"
    [finetune]="expts/finetune.py"
    [generate]="expts/generate.py"
)
declare -A EXPERIMENT_SCRIPTS=(
    [embed]="scripts/embed.sh"
    [classify]="scripts/classify.sh"
    [classify2]="scripts/classify2.sh"
    [condense]="scripts/condense.sh"
    [finetune]="scripts/finetune.sh"
    [generate]="scripts/generate.sh"
)

menu() {
    local title=$1
    shift
    local -a options=("$@")

    echo "" >&2
    echo "=== $title ===" >&2
    for i in "${!options[@]}"; do
        [[ $i -eq $((${#options[@]}-1)) ]] && continue
        echo "[$((i+1))] ${options[$i]}" >&2
    done
    echo "[0] ${options[-1]}" >&2
    echo "" >&2
    read -p "Choose [0-$((${#options[@]}-1))] (default 1): " choice
    choice=${choice:-1}

    if [[ $choice -eq 0 ]]; then
        echo "${options[-1]}"
    elif [[ $choice -ge 1 && $choice -le $((${#options[@]}-1)) ]]; then
        echo "${options[$((choice-1))]}"
    fi
}

run_local() {
    local experiment=$1
    local py_kwargs=$2
    source "$REPO/.venv/bin/activate"
    python -u "$REPO/${EXPERIMENTS[$experiment]}" $py_kwargs
}

run_slurm() {
    local experiment=$1
    local slurm_kwargs=$2
    local py_kwargs=$3

    sbatch \
        --output="$REPO/slurm/%j.out" \
        --error="$REPO/slurm/%j.err" \
        $slurm_kwargs \
        "$REPO/${EXPERIMENT_SCRIPTS[$experiment]}" \
        $py_kwargs

    echo "Job submitted!"
    sleep 1
}

_pick_preset() {
    local prefix=$1 title=$2
    local -a names=() vals=()
    while IFS='=' read -r key val; do
        local label="${key#${prefix}}"
        names+=("${label#*_}")
        vals+=("${val//\"/}")
    done < <(compgen -v | grep "^${prefix}" | sort -V | while read -r v; do echo "$v=${!v}"; done)
    local choice
    choice=$(menu "$title" "${names[@]}" "Back")
    [[ -z "$choice" || "$choice" == "Back" ]] && return 1
    for i in "${!names[@]}"; do
        [[ "${names[$i]}" == "$choice" ]] && echo "${vals[$i]}" && return 0
    done
}

main() {
    while true; do
        local -a experiments=("${EXPERIMENT_NAMES[@]}" "Exit")

        exp=$(menu "TextDD Experiments" "${experiments[@]}")
        [[ "$exp" == "Exit" ]] && exit 0
        [[ -z "$exp" ]] && continue

        mode=$(menu "$exp - Run Mode" "SLURM" "Local" "Back")
        [[ -z "$mode" || "$mode" == "Back" ]] && continue

        if [[ "$mode" == "Local" ]]; then
            sed -n '/^# Args:/,/^[^#]/{ /^[^#]/d; p }' "$REPO/${EXPERIMENT_SCRIPTS[$exp]}" >&2
            read -p "Python kwargs: " py_kwargs
            run_local "$exp" "$py_kwargs"
        else
            local spec_kwargs dur_kwargs
            spec_kwargs=$(_pick_preset "SPEC_" "SLURM Spec") || continue
            dur_kwargs=$(_pick_preset "DUR_" "SLURM Duration") || continue
            local slurm_kwargs="$spec_kwargs $dur_kwargs"

            sed -n '/^# Args:/,/^[^#]/{ /^[^#]/d; p }' "$REPO/${EXPERIMENT_SCRIPTS[$exp]}" >&2
            read -p "Python kwargs: " py_kwargs
            echo "" >&2
            echo "sbatch \\" >&2
            echo "    --output=$REPO/slurm/%j.out \\" >&2
            echo "    --error=$REPO/slurm/%j.err \\" >&2
            for kw in $slurm_kwargs; do echo "    $kw \\" >&2; done
            echo "    $REPO/${EXPERIMENT_SCRIPTS[$exp]} \\" >&2
            echo "    $py_kwargs" >&2
            echo "" >&2
            read -p "Submit? [Y/n]: " confirm
            [[ "$confirm" =~ ^[Nn] ]] && continue
            run_slurm "$exp" "$slurm_kwargs" "$py_kwargs"
        fi

        read -p "Press Enter to continue..."
    done
}

main
