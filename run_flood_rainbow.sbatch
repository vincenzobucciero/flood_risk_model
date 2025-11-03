#!/bin/bash
#SBATCH -J runoff1h
#SBATCH -p gpu                    # Cambia se serve: short / standard / long
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/runoff1h-%j.out
#SBATCH -e logs/runoff1h-%j.err

set -euo pipefail

WORKDIR="/home/vbucciero/projects/flood_risk_model"
cd "$WORKDIR"

mkdir -p logs outputs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

ENV_PATH="$HOME/.local/share/mamba/envs/flood"

echo "=== JOB INFO ==="
echo "Workdir: $WORKDIR"
echo "Env path: $ENV_PATH"
echo "CPUs: $OMP_NUM_THREADS"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "================"

# misura tempo totale
START_TIME=$(date +%s)

# Esegui con micromamba (preferito)
if command -v micromamba >/dev/null 2>&1; then
  echo "[INFO] Esecuzione con micromamba..." | tee >(cat >&2)
  micromamba run -p "$ENV_PATH" python main_run_1h.py 2>&1 | tee logs/runoff1h-${SLURM_JOB_ID}.live.log
else
  echo "[WARN] Micromamba non trovato, provo con conda/mamba..." | tee >(cat >&2)
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python main_run_1h.py 2>&1 | tee logs/runoff1h-${SLURM_JOB_ID}.live.log
  elif command -v mamba >/dev/null 2>&1; then
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python main_run_1h.py 2>&1 | tee logs/runoff1h-${SLURM_JOB_ID}.live.log
  else
    echo "[ERROR] Nessuna versione di (micro)mamba o conda trovata." >&2
    exit 1
  fi
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "================"
echo "Fine job: $(date)"
echo "Tempo totale: ${ELAPSED} secondi (~$(echo "$ELAPSED / 60" | bc) min)"
echo "================"
