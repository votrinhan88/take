#!/bin/bash
# Interactive TUI for launching experiments with slurm/python kwargs

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/.env"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Experiments
declare -A EXPERIMENTS=(
    [embed]="expts/embed.py"
    [classify]="expts/classify.py"
    [classify2]="expts/classify2.py"
    [condense]="expts/condense.py"
    [finetune]="expts/finetune.py"
    [generate]="expts/generate.py"
)

show_menu() {
    clear
    echo -e "${BLUE}=== TextDD Experiment Launcher ===${NC}\n"
    echo "Select experiment:"
    local i=1
    for name in "${!PIPELINES[@]}"; do
        echo "  $i) $name"
        ((i++))
    done
    echo "  0) Exit"
    echo ""
}

get_pipeline_choice() {
    local choice
    read -p "Enter choice: " choice

    local i=1
    for name in "${!EXPERIMENTS[@]}"; do
        if [[ $choice -eq $i ]]; then
            echo "$name"
            return 0
        fi
        ((i++))
    done

    if [[ $choice -eq 0 ]]; then
        echo "exit"
        return 0
    fi

    echo ""
}

show_run_mode() {
    local pipeline=$1
    clear
    echo -e "${BLUE}=== $pipeline ===${NC}\n"
    echo "Run mode:"
    echo "  1) Local (python)"
    echo "  2) SLURM (sbatch)"
    echo "  0) Back"
    echo ""
}

get_run_mode() {
    read -p "Enter choice: " choice
    echo "$choice"
}

get_kwargs() {
    local mode=$1  # "local" or "slurm"
    local prompt=""

    if [[ "$mode" == "slurm" ]]; then
        prompt="Enter SLURM kwargs (e.g., --gres=gpu:1 --time=01:00:00): "
    else
        prompt="Enter Python kwargs (e.g., --config=clf-logistic-tfidf-agnews): "
    fi

    read -p "$prompt" kwargs
    echo "$kwargs"
}

get_python_kwargs() {
    read -p "Enter Python kwargs (e.g., --config=clf-logistic-tfidf-agnews): " kwargs
    echo "$kwargs"
}

run_local() {
    local experiment=$1
    local py_kwargs=$2

    echo -e "\n${GREEN}Running locally...${NC}"
    source "$REPO/.venv/bin/activate"
    python "$REPO/${EXPERIMENTS[$experiment]}" $py_kwargs
}

run_slurm() {
    local experiment=$1
    local slurm_kwargs=$2
    local py_kwargs=$3

    echo -e "\n${GREEN}Submitting to SLURM...${NC}"

    sbatch \
        --output="$REPO/slurm/%j.out" \
        --error="$REPO/slurm/%j.err" \
        $slurm_kwargs \
        --wrap="source $REPO/.env && source $REPO/.venv/bin/activate && python $REPO/${EXPERIMENTS[$experiment]} $py_kwargs"

    echo -e "${GREEN}Job submitted!${NC}"
    sleep 2
}

main() {
    while true; do
        show_menu
        experiment=$(get_pipeline_choice)

        if [[ "$experiment" == "exit" ]]; then
            echo "Exiting..."
            exit 0
        fi

        if [[ -z "$experiment" ]]; then
            continue
        fi

        while true; do
            show_run_mode "$experiment"
            mode=$(get_run_mode)

            case $mode in
                1)
                    py_kwargs=$(get_python_kwargs)
                    run_local "$experiment" "$py_kwargs"
                    read -p "Press Enter to continue..."
                    break
                    ;;
                2)
                    slurm_kwargs=$(get_kwargs "slurm")
                    py_kwargs=$(get_python_kwargs)
                    run_slurm "$experiment" "$slurm_kwargs" "$py_kwargs"
                    read -p "Press Enter to continue..."
                    break
                    ;;
                0)
                    break
                    ;;
                *)
                    echo -e "${RED}Invalid choice${NC}"
                    sleep 1
                    ;;
            esac
        done
    done
}

main
