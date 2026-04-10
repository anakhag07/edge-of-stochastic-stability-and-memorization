#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=eoss-fork-v2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-1

# =============================================================================
# Fork-based EoS intervention using v2 prototypes (both-class x-outliers).
#
# Identical to fork_intervention.slurm except:
#   - Uses v2 prototype path (20260404_1350_54) with symmetric x-outliers
#   - Logs to separate W&B project to avoid mixing with original results
#
# Usage:
#   # Phase 1: run all 11 baselines
#   for S in 1111 1234 3333 3902 4321 5555 5678 6543 7070 7777 9999; do
#       sbatch --array=0 --export=ALL,SEED=$S fork_intervention_v2proto.slurm
#   done
#
#   # Phase 2: fork from baseline checkpoint
#   sbatch --array=1 --export=ALL,SEED=1234,BASELINE_RUN_ID=<id>,TSTAR_STEP=<step> fork_intervention_v2proto.slurm
# =============================================================================

# ---------- CLI parameters with defaults ----------
SEED=${SEED:-1234}
CLASSES=${CLASSES:-"1 9"}
LR=${LR:-0.01}
STEPS=${STEPS:-100000}
MODEL=${MODEL:-mlp}
LOSS=${LOSS:-mse}

# ---------- Drop multiplier: default 0.1, override to 0.5 for classes 3 5 ----------
if [[ -z "$DROP_MULT" ]]; then
    if [[ "$CLASSES" == "3 5" ]]; then
        DROP_MULT=0.5
    else
        DROP_MULT=0.1
    fi
fi

# ---------- v2 prototype path (both-class x-outliers) ----------
PROTO="$RESULTS/plaintext/cifar10_2cls_mlp/20260404_1350_54_input_prototypes/input_prototypes"

# ---------- WANDB config (separate project) ----------

# ---------- Use large batch to guarantee full-batch GD ----------
BATCH=100000

# ---------- Common training flags ----------
COMMON_FLAGS="
    --dataset      cifar10_2cls
    --classes      $CLASSES
    --num-data     10000
    --model        $MODEL
    --loss         $LOSS
    --batch        $BATCH
    --init-scale   0.2
    --dataset-seed 888
    --init-seed    $SEED
    --train-input-prototypes from:$PROTO
    --input-prototypes-count 50
    --lambdamax
    --train-input-x-outliers 25
    --train-input-y-outliers 25
    --train-input-inliers    25
    --train-input-boundary   25
    --steps        $STEPS
"

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
    0)
        # ── Phase 1: Baseline (EoS, full LR) ────────────────────────────
        NAME="fork_baseline_seed${SEED}_lr${LR}"
        echo "=============================================="
        echo "PHASE 1: BASELINE (v2 proto)"
        echo "SEED=${SEED}  LR=${LR}  STEPS=${STEPS}"
        echo "NAME=${NAME}"
        echo "WANDB_PROJECT: ${WANDB_PROJECT}"
        echo "PROTO: ${PROTO}"
        echo "=============================================="

        python training.py \
            $COMMON_FLAGS \
            --lr           "$LR" \
            --wandb-name   "$NAME"
        ;;

    1)
        # ── Phase 2: Exit-EoS fork from baseline checkpoint ─────────────
        if [[ -z "$BASELINE_RUN_ID" || -z "$TSTAR_STEP" ]]; then
            echo "ERROR: Phase 2 requires BASELINE_RUN_ID and TSTAR_STEP."
            echo "  Usage: sbatch --array=1 --export=ALL,SEED=$SEED,BASELINE_RUN_ID=<id>,TSTAR_STEP=<step> fork_intervention_v2proto.slurm"
            exit 1
        fi

        NEW_LR=$(awk -v lr="$LR" -v m="$DROP_MULT" 'BEGIN { printf "%.6g", lr * m }')
        NAME="fork_exit_eos_seed${SEED}_lr${LR}_drop${DROP_MULT}_from_step${TSTAR_STEP}"

        echo "=============================================="
        echo "PHASE 2: EXIT-EOS FORK (v2 proto)"
        echo "SEED=${SEED}  LR=${LR} -> ${NEW_LR}  DROP_MULT=${DROP_MULT}"
        echo "BASELINE_RUN_ID=${BASELINE_RUN_ID}  TSTAR_STEP=${TSTAR_STEP}"
        echo "NAME=${NAME}"
        echo "WANDB_PROJECT: ${WANDB_PROJECT}"
        echo "=============================================="

        python training.py \
            $COMMON_FLAGS \
            --lr           "$LR" \
            --lr-drop-at-step "$TSTAR_STEP" \
            --lr-drop-to   "$NEW_LR" \
            --cont-run-id  "$BASELINE_RUN_ID" \
            --cont-step    "$TSTAR_STEP" \
            --wandb-name   "$NAME"
        ;;

    *)
        echo "Unknown array index: $IDX"; exit 1
        ;;
esac
