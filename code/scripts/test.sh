#!/bin/bash
#SBATCH --job-name=tgppo_test_final
#SBATCH --partition=himem
#SBATCH --qos=himem-cpu
#SBATCH --cpus-per-task=66              # parallel eval workers
#SBATCH --mem=300G                      # increase if SCIP uses more
#SBATCH --time=36:00:00                 # 325 runs at 1h cutoff; parallelized
#SBATCH --output=output/hpclogs/%x-%j.out
#SBATCH --error=output/hpclogs/%x-%j.err
#SBATCH --account=OPT_ML-FWG2KAAYBVI-DEFAULT-cpu

# -------- Modules (CPU run) --------
module load GCCcore/10.2.0
module load GMP/6.2.0-GCCcore-10.2.0

# -------- Paths --------
PERSIST_ROOT="output"
DIR_HPCLOGS="${PERSIST_ROOT}/hpclogs"
DIR_TEST_LOGS="${PERSIST_ROOT}/logs/testing_final"

# Fast local scratch for tmp logs
LOGS_TMP="${SLURM_TMPDIR:-/tmp}/tgppo_test_${SLURM_JOB_ID}"

mkdir -p "$DIR_HPCLOGS" "$DIR_TEST_LOGS" "$LOGS_TMP"

# -------- Env hygiene --------
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


# -------- Data & model --------
TEST_INST_DIR="dataset/tbrant/test_instances"
INFO_PKL="dataset/tbrant/cutoff_test.pkl"   # or testing info dict if different
MODEL_PATH="model/tgppo_final_model.pt"


python - <<'PY'
import torch, sys, pprint
ckpt = torch.load("model/tgppo_final_model.pt", map_location="cpu")
print("Checkoint Here is it")
pprint.pprint(list(ckpt.keys()))
PY

# -------- Script & settings --------
SCRIPT="test.py"
TIME_LIMIT=3600
PER_JOB_TIMEOUT=3900
SEEDS="0 1 2 3 4"
MAX_WORKERS="$SLURM_CPUS_PER_TASK"
SHIFT=100

set -e

# IMPORTANT: The architecture in testing.py's build_models() must match the saved model.
# If you changed hidden_dim / layers / heads during training, edit testing.py accordingly
# before running this job.

python "$SCRIPT" \
  --instances_dir "$TEST_INST_DIR" \
  --instances_info_dict "$INFO_PKL" \
  --model_path "$MODEL_PATH" \
  --logs_dir "$LOGS_TMP" \
  --time_limit "$TIME_LIMIT" \
  --per_job_timeout "$PER_JOB_TIMEOUT" \
  --scip_setting sandbox \
  --seeds $SEEDS \
  --max_workers "$MAX_WORKERS" \
  --shift "$SHIFT"

echo "Syncing results to $DIR_TEST_LOGS ..."
rsync -a "$LOGS_TMP"/ "$DIR_TEST_LOGS"/

echo "Done."
