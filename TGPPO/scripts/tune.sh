#!/bin/bash
#SBATCH --job-name=tg_ppo_nestedcv
#SBATCH --partition=himem
#SBATCH --qos=himem-cpu
#SBATCH --cpus-per-task=40              # parallel workers used inside each phase
#SBATCH --mem=300G                      # SCIP can be memory-hungry; adjust as needed
#SBATCH --time=36:00:00
#SBATCH --output=output/hpclogs/%x-%j.out
#SBATCH --error=output/hpclogs/%x-%j.err
#SBATCH --account=OPT_ML-FWG2KAAYBVI-DEFAULT-cpu

# ---- Modules (GPU not required; keep CUDA only if needed elsewhere) ----
module load GCCcore/10.2.0
module load GMP/6.2.0-GCCcore-10.2.0


# -------------------- Directories --------------------
PERSIST_LOG_ROOT="output/logs/nested_cv"
DIR_HPCLOGS="output/hpclogs"
OPTUNA_DIR="output/optuna"

# Use fast local scratch for rollouts & logs, then copy back
LOGS_TMP="${SLURM_TMPDIR:-/tmp}/tgppo_nestedcv_${SLURM_JOB_ID}"
mkdir -p "$DIR_HPCLOGS" "$OPTUNA_DIR" "$PERSIST_LOG_ROOT" "$LOGS_TMP"

# -------------------- Environment hygiene --------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Host: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
python - <<'PY'
import psutil, platform, os
print("RAM total GB:", round(psutil.virtual_memory().total/1024**3,2))
print("Python:", platform.python_version())
print("TMPDIR:", os.environ.get("SLURM_TMPDIR") or "/tmp")
PY

# -------------------- Paths to data --------------------
INST_DIR="dataset/tbrant/train_instances"          # single directory with the 25 instances
INFO_PKL="dataset/tbrant/train_instances_infos.pkl"

# -------------------- Script & settings --------------------
SCRIPT="tune.py"

TIME_LIMIT=900
PER_JOB_TIMEOUT=900
OUTER_SEEDS="0 1"
INNER_SEEDS="0"
INNER_FOLDS=2
TRAIN_ITERS_INNER=3
TRAIN_ITERS_REFIT=6
MAX_WORKERS="$SLURM_CPUS_PER_TASK"
N_TRIALS_INNER=60

STORAGE_INNER="${OPTUNA_DIR}/optuna_inner.db"

# -------------------- Run --------------------
set -e
python "$SCRIPT" \
  --instances_dir "$INST_DIR" \
  --instances_info_dict "$INFO_PKL" \
  --logs_dir "$LOGS_TMP" \
  --time_limit "$TIME_LIMIT" \
  --per_job_timeout "$PER_JOB_TIMEOUT" \
  --scip_setting sandbox \
  --outer_seeds $OUTER_SEEDS \
  --inner_seeds $INNER_SEEDS \
  --inner_folds $INNER_FOLDS \
  --train_iterations_inner $TRAIN_ITERS_INNER \
  --train_iterations_refit $TRAIN_ITERS_REFIT \
  --max_workers "$MAX_WORKERS" \
  --sampler tpe \
  --pruner median \
  --n_trials_inner "$N_TRIALS_INNER" \
  --storage_inner "$STORAGE_INNER"

# -------------------- Persist results --------------------
echo "Syncing results to $PERSIST_LOG_ROOT ..."
rsync -a "$LOGS_TMP"/ "$PERSIST_LOG_ROOT"/

echo "Done."