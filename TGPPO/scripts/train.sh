#!/bin/bash
#SBATCH --job-name=tgppo_train_final
#SBATCH --partition=himem
#SBATCH --qos=himem-cpu
#SBATCH --cpus-per-task=40          # parallel workers inside training.py
#SBATCH --mem=600G                  # adjust to your node / instance sizes
#SBATCH --time=36:00:00
#SBATCH --output=output/hpclogs/%x-%j.out
#SBATCH --error=output/hpclogs/%x-%j.err
#SBATCH --account=OPT_ML-FWG2KAAYBVI-DEFAULT-cpu

# ---------- Modules (CPU build; drop CUDA unless you really need it) ----------
module load GCCcore/10.2.0
module load GMP/6.2.0-GCCcore-10.2.0
# If SciPy warns about NumPy mismatch, fix in your env:
# conda install -n ictai2024 "scipy==1.10.*"  # works with numpy 1.24

# ---------- Paths ----------
PERSIST_ROOT="output"                                  # persistent project root
DIR_HPCLOGS="${PERSIST_ROOT}/hpclogs"
DIR_LOGS="${PERSIST_ROOT}/logs/final_training"
DIR_MODELS="${PERSIST_ROOT}/models"
DIR_CKPTS="${PERSIST_ROOT}/checkpoints"

# Fast local scratch for tmp trajectories & intermediate logs
LOGS_TMP="${SLURM_TMPDIR:-/tmp}/tgppo_train_${SLURM_JOB_ID}"

mkdir -p "$DIR_HPCLOGS" "$DIR_LOGS" "$DIR_MODELS" "$DIR_CKPTS" "$LOGS_TMP"

# ---------- Env hygiene ----------
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
print("RAM total GB:", round(psutil.virtual_memory().total/1024**3, 2))
print("Python:", platform.python_version())
print("TMPDIR:", os.environ.get("SLURM_TMPDIR") or "/tmp")
PY

# ---------- Data ----------
INST_DIR="dataset/debuging/train_instances"                 # all 25 instances
INFO_PKL="dataset/debuging/train_instances_infos.pkl"
BEST_HP_JSON="params/best_hparams_aggregate.json"  # or your chosen JSON

# ---------- Training config ----------
SCRIPT="train.py"
TIME_LIMIT=600
PER_JOB_TIMEOUT=1200
SEEDS="0"                      # training seeds for data collection
VAL_INST_DIR=""                    # optional separate validation dir; leave empty to skip
VAL_SEEDS="0"
TRAIN_ITERS=12
CKPT_EVERY=2
MAX_WORKERS="$SLURM_CPUS_PER_TASK"

OUT_MODEL="${DIR_MODELS}/tgppo_final_model.pt"

set -e

# ---------- Run ----------
python "$SCRIPT" \
  --instances_dir "$INST_DIR" \
  --instances_info_dict "$INFO_PKL" \
  --logs_dir "$LOGS_TMP" \
  --best_params_json "$BEST_HP_JSON" \
  --output_model "$OUT_MODEL" \
  --checkpoint_dir "$DIR_CKPTS" \
  --time_limit "$TIME_LIMIT" \
  --per_job_timeout "$PER_JOB_TIMEOUT" \
  --scip_setting sandbox \
  --seeds $SEEDS \
  --train_iterations "$TRAIN_ITERS" \
  --checkpoint_every "$CKPT_EVERY" \
  --max_workers "$MAX_WORKERS" \
  $( [ -n "$VAL_INST_DIR" ] && echo --val_instances_dir "$VAL_INST_DIR" ) \
  $( [ -n "$VAL_SEEDS" ] && echo --val_seeds $VAL_SEEDS )

# ---------- Persist logs ----------
echo "Syncing logs from ${LOGS_TMP} -> ${DIR_LOGS}"
rsync -a "$LOGS_TMP"/ "$DIR_LOGS"/

echo "Done."
