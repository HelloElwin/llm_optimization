#!/bin/bash
# =============================================================================
# Best-of-N Scaling Experiment Script
# =============================================================================
# This script runs BoN experiments to study how overoptimization scales with
# different reward model (RM) sizes and policy sizes.
#
# Experiment Design:
#   - Sweep RM sizes: 44M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
#   - Multiple seeds per RM size (no ensembles, individual RMs only)
#   - Default policy: 1.4B (from tlc4418/gold_labelled_gens)
#
# Prerequisites:
#   1. Train reward models for each size with multiple seeds:
#      accelerate launch src/reward_modeling/training/trainer_rm.py \
#        --configs defaults_rm rm-pythia-{size} --rng_seed {seed}
#
#   2. Ensure gold-labelled generations dataset is accessible
#
# Usage:
#   ./runs/run_bon_scaling_sweep.sh [options]
#
# Options:
#   --rm-sizes      Comma-separated list of RM sizes (default: all)
#   --seeds         Comma-separated list of seeds (default: 1,2,3)
#   --output-base   Base output directory (default: bon_scaling_sweep)
#   --dry-run       Print commands without executing
#   --skip-missing  Skip RMs that don't exist instead of failing
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default RM sizes (all Pythia v1 models)
DEFAULT_RM_SIZES="44m,160m,410m,1b,1.4b,2.8b,6.9b,12b"

# Default seeds (3 seeds for statistical significance)
DEFAULT_SEEDS="1,2,3"

# Output directory base
OUTPUT_BASE="bon_scaling_sweep"

# Gold-labelled generations dataset (1.4B policy)
GOLD_DATASET="tlc4418/gold_labelled_gens"

# BoN sampling parameters
BIG_N=12600
SAMPLE_NS="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,6144,8192,12500"

# Flags
DRY_RUN=false
SKIP_MISSING=false

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
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-missing)
            SKIP_MISSING=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --rm-sizes      Comma-separated list of RM sizes (default: $DEFAULT_RM_SIZES)"
            echo "  --seeds         Comma-separated list of seeds (default: $DEFAULT_SEEDS)"
            echo "  --output-base   Base output directory (default: bon_scaling_sweep)"
            echo "  --dry-run       Print commands without executing"
            echo "  --skip-missing  Skip RMs that don't exist instead of failing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
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

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Get RM path for a given size
get_rm_path() {
    local size=$1
    # Handle 70m -> 44m naming convention
    if [[ "$size" == "70m" ]]; then
        echo "models/rm-pythia-44m_seed{seed}"
    elif [[ "$size" == "44m" ]]; then
        echo "models/rm-pythia-44m_seed{seed}"
    else
        echo "models/rm-pythia-${size}_seed{seed}"
    fi
}

# Check if RM exists for a given size and seed
check_rm_exists() {
    local size=$1
    local seed=$2
    local rm_path

    if [[ "$size" == "70m" || "$size" == "44m" ]]; then
        rm_path="models/rm-pythia-44m_seed${seed}"
    else
        rm_path="models/rm-pythia-${size}_seed${seed}"
    fi

    if [[ -d "$rm_path" ]]; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

log_info "=============================================="
log_info "Best-of-N Scaling Sweep Experiment"
log_info "=============================================="
log_info "RM Sizes: $RM_SIZES"
log_info "Seeds: $SEEDS"
log_info "Output Base: $OUTPUT_BASE"
log_info "Gold Dataset: $GOLD_DATASET"
log_info "Dry Run: $DRY_RUN"
log_info "=============================================="

# Create output directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "runs/${OUTPUT_DIR}"
    log_info "Created output directory: runs/${OUTPUT_DIR}"
fi

# Convert comma-separated lists to arrays
IFS=',' read -ra RM_SIZE_ARRAY <<< "$RM_SIZES"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Track experiment status
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# =============================================================================
# Run BoN for each RM size and seed
# =============================================================================

for rm_size in "${RM_SIZE_ARRAY[@]}"; do
    log_info "----------------------------------------------"
    log_info "Processing RM size: ${rm_size}"
    log_info "----------------------------------------------"

    for seed in "${SEED_ARRAY[@]}"; do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

        # Check if RM exists
        if ! check_rm_exists "$rm_size" "$seed"; then
            if [[ "$SKIP_MISSING" == true ]]; then
                log_warn "RM not found for size=${rm_size}, seed=${seed}. Skipping..."
                SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
                continue
            else
                log_error "RM not found for size=${rm_size}, seed=${seed}"
                log_error "Train it first with:"
                log_error "  accelerate launch src/reward_modeling/training/trainer_rm.py \\"
                log_error "    --configs defaults_rm rm-pythia-${rm_size} --rng_seed ${seed}"
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
                continue
            fi
        fi

        # Get RM path template
        rm_path=$(get_rm_path "$rm_size")

        # Construct output name
        exp_output="${OUTPUT_DIR}/rm_${rm_size}"

        log_info "Running BoN: RM=${rm_size}, seed=${seed}"

        # Construct command
        CMD="python src/bon/run_bon_pipeline.py ${rm_path} \
            -o ${exp_output} \
            --gold-labelled-generations ${GOLD_DATASET} \
            --big-n ${BIG_N} \
            --sample-ns ${SAMPLE_NS} \
            --seeds ${seed}"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] $CMD"
        else
            log_info "Executing: $CMD"
            if eval "$CMD"; then
                COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
                log_info "Completed: RM=${rm_size}, seed=${seed}"
            else
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
                log_error "Failed: RM=${rm_size}, seed=${seed}"
            fi
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================

log_info "=============================================="
log_info "Experiment Summary"
log_info "=============================================="
log_info "Total experiments: $TOTAL_EXPERIMENTS"
log_info "Completed: $COMPLETED_EXPERIMENTS"
log_info "Skipped: $SKIPPED_EXPERIMENTS"
log_info "Failed: $FAILED_EXPERIMENTS"

if [[ "$DRY_RUN" == false ]]; then
    log_info "Results saved to: runs/${OUTPUT_DIR}/"

    # Save experiment configuration
    cat > "runs/${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "bon_scaling_sweep",
    "timestamp": "${TIMESTAMP}",
    "rm_sizes": "${RM_SIZES}",
    "seeds": "${SEEDS}",
    "gold_dataset": "${GOLD_DATASET}",
    "big_n": ${BIG_N},
    "sample_ns": "${SAMPLE_NS}",
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "completed": ${COMPLETED_EXPERIMENTS},
    "skipped": ${SKIPPED_EXPERIMENTS},
    "failed": ${FAILED_EXPERIMENTS}
}
EOF
    log_info "Saved experiment config to: runs/${OUTPUT_DIR}/experiment_config.json"
fi

log_info "=============================================="
log_info "Done!"
log_info "=============================================="

# Exit with error if any experiments failed
if [[ $FAILED_EXPERIMENTS -gt 0 ]]; then
    exit 1
fi
