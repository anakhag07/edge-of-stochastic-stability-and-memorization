#!/bin/bash
# =============================================================================
# Helper: read t* from a completed baseline run and submit the exit-EoS fork.
#
# Usage:
#   bash submit_fork.sh <baseline_wandb_run_id> [SEED] [DROP_MULT] [CLASSES]
#
# Example:
#   # After baseline finishes, get run_id from wandb or logs:
#   bash submit_fork.sh abc123def 1234
#   bash submit_fork.sh abc123def 1234 0.5 "3 5"
#
# Full workflow:
#   # 1. Submit baseline for each seed
#   for SEED in 1234 3902 5678 8312; do
#       sbatch --array=0 --export=ALL,SEED=$SEED fork_intervention.slurm
#   done
#
#   # 2. After baselines finish, sync wandb and read t* for each:
#   cd $WANDB_DIR && wandb sync --sync-all --include-offline --mark-synced --no-include-synced
#
#   # 3. Submit forks (run_ids from wandb sync output or wandb UI):
#   bash submit_fork.sh <run_id_seed1234> 1234
#   bash submit_fork.sh <run_id_seed3902> 3902
#   bash submit_fork.sh <run_id_seed5678> 5678
#   bash submit_fork.sh <run_id_seed8312> 8312
# =============================================================================

set -e

BASELINE_RUN_ID=${1:?  "Usage: bash submit_fork.sh <baseline_run_id> [SEED] [CLASSES] [DROP_MULT]"}
SEED=${2:-1234}
CLASSES=${3:-"1 9"}

# Default DROP_MULT: 0.1, but 0.5 for classes 3 5
if [[ -n "$4" ]]; then
    DROP_MULT=$4
elif [[ "$CLASSES" == "3 5" ]]; then
    DROP_MULT=0.5
else
    DROP_MULT=0.1
fi

echo "Querying t* from baseline run: $BASELINE_RUN_ID"

# Read t* step from W&B summary
TSTAR_STEP=$(OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n eoss python -W ignore -c "
import wandb, json, sys
api = wandb.Api()
for proj in [
    'shaunakwag-massachusetts-institute-of-technology/fork-eoss-v2proto-full_gd-mlp-mse-cifar10_2cls',
    'shaunakwag-massachusetts-institute-of-technology/eoss-fork-full_gd-mlp-mse-cifar10_2cls',
    'shaunakwag-massachusetts-institute-of-technology/debugging-eoss-full_gd-mlp-mse-cifar10_2cls_cls3_cls5',
]:
    try:
        run = api.run(f'{proj}/{sys.argv[1]}')
        summ = dict(run.summary)
        t_star = summ.get('t_star', None)
        if t_star is not None:
            print(int(t_star))
            break
    except:
        continue
" "$BASELINE_RUN_ID" 2>/dev/null || true)

if [[ -z "$TSTAR_STEP" || "$TSTAR_STEP" == "NOT_FOUND" ]]; then
    echo "ERROR: Could not read t_star from W&B summary for run $BASELINE_RUN_ID"
    echo ""
    echo "You can manually specify the step:"
    echo "  sbatch --array=1 --export=ALL,SEED=$SEED,BASELINE_RUN_ID=$BASELINE_RUN_ID,TSTAR_STEP=<step>,DROP_MULT=$DROP_MULT fork_intervention.slurm"
    exit 1
fi

echo "Found t* = step $TSTAR_STEP"
echo ""
echo "Submitting exit-EoS fork:"
echo "  SEED=$SEED  DROP_MULT=$DROP_MULT  BASELINE=$BASELINE_RUN_ID  TSTAR=$TSTAR_STEP"
echo ""

cd "$(dirname "$0")"

sbatch --array=1 \
    --export=ALL,SEED=$SEED,BASELINE_RUN_ID=$BASELINE_RUN_ID,TSTAR_STEP=$TSTAR_STEP,DROP_MULT=$DROP_MULT,CLASSES="$CLASSES" \
    fork_intervention_v2proto.slurm
