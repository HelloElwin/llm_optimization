#!/bin/bash
# =============================================================================
# Train Reward Models for Scaling Experiments
# =============================================================================
# This script trains reward models of different sizes with multiple seeds,
# preparing them for the BoN scaling sweep experiment.
#
# Usage:
#   ./runs/train_rms_for_scaling.sh [options]
#
# Options:
#   --rm-sizes      Comma-separated list of RM sizes (default: all)
#   --seeds         Comma-separated list of seeds (default: 1,2,3)
#   --use-lora      Use LoRA for training (faster, less memory)
#   --dry-run       Print commands without executing
#   --skip-existing Skip training if model already exists
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Default RM sizes (all Pythia v1 models)
# Note: 44m refers to the RM from 70m base model
DEFAULT_RM_SIZES="44m,160m,410m,1b,1.4b,2.8b,6.9b,12b"

# Default seeds
DEFAULT_SEEDS="1,2,3"

# Flags
DRY_RUN=false
USE_LORA=false
SKIP_EXISTING=false

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --rm-sizes)
            RM_SIZES="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --use-lora)
            USE_LORA=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --rm-sizes      Comma-separated list of RM sizes (default: $DEFAULT_RM_SIZES)"
            echo "  --seeds         Comma-separated list of seeds (default: $DEFAULT_SEEDS)"
            echo "  --use-lora      Use LoRA for training (faster, less memory)"
            echo "  --dry-run       Print commands without executing"
            echo "  --skip-existing Skip training if model already exists"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
RM_SIZES="${RM_SIZES:-$DEFAULT_RM_SIZES}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Get config name for RM size
get_rm_config() {
    local size=$1
    local use_lora=$2

    # Handle 44m -> rm-pythia-44m
    if [[ "$size" == "70m" ]]; then
        size="44m"
    fi

    if [[ "$use_lora" == true ]]; then
        echo "rm-pythia-${size}-lora"
    else
        echo "rm-pythia-${size}"
    fi
}

# Get expected output path
get_output_path() {
    local size=$1
    local seed=$2
    local use_lora=$3

    if [[ "$size" == "70m" ]]; then
        size="44m"
    fi

    if [[ "$use_lora" == true ]]; then
        echo "models/rm-pythia-${size}-lora_seed${seed}"
    else
        echo "models/rm-pythia-${size}_seed${seed}"
    fi
}

# Check if model exists
check_exists() {
    local path=$1
    [[ -d "$path" ]]
}

# =============================================================================
# Main Execution
# =============================================================================

log_info "=============================================="
log_info "Train Reward Models for Scaling Experiments"
log_info "=============================================="
log_info "RM Sizes: $RM_SIZES"
log_info "Seeds: $SEEDS"
log_info "Use LoRA: $USE_LORA"
log_info "Dry Run: $DRY_RUN"
log_info "Skip Existing: $SKIP_EXISTING"
log_info "=============================================="

# Convert to arrays
IFS=',' read -ra RM_SIZE_ARRAY <<< "$RM_SIZES"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Track status
TOTAL=0
TRAINED=0
SKIPPED=0
FAILED=0

for rm_size in "${RM_SIZE_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        TOTAL=$((TOTAL + 1))

        config_name=$(get_rm_config "$rm_size" "$USE_LORA")
        output_path=$(get_output_path "$rm_size" "$seed" "$USE_LORA")

        log_info "----------------------------------------------"
        log_info "RM Size: ${rm_size}, Seed: ${seed}"
        log_info "Config: ${config_name}"
        log_info "Output: ${output_path}"

        # Check if exists
        if [[ "$SKIP_EXISTING" == true ]] && check_exists "$output_path"; then
            log_info "Model already exists. Skipping..."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Construct command
        CMD="accelerate launch src/reward_modeling/training/trainer_rm.py \
            --configs defaults_rm ${config_name} \
            --rng_seed ${seed}"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] $CMD"
            TRAINED=$((TRAINED + 1))
        else
            log_info "Executing: $CMD"
            if eval "$CMD"; then
                TRAINED=$((TRAINED + 1))
                log_info "Successfully trained: ${config_name} seed ${seed}"
            else
                FAILED=$((FAILED + 1))
                log_warn "Failed to train: ${config_name} seed ${seed}"
            fi
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================

log_info "=============================================="
log_info "Training Summary"
log_info "=============================================="
log_info "Total: $TOTAL"
log_info "Trained: $TRAINED"
log_info "Skipped: $SKIPPED"
log_info "Failed: $FAILED"
log_info "=============================================="

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
