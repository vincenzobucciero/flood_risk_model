#!/bin/bash
#SBATCH -J runoff1h
#SBATCH -p gpu                  # <-- cambia la partition se diversa (es: short, standard, long)
#SBATCH -c 8                    # CPU per task
#SBATCH --mem=32G               # RAM
#SBATCH -t 02:00:00             # walltime
#SBATCH -o logs/runoff1h-%j.out
#SBATCH -e logs/runoff1h-%j.err

set -euo pipefail

# Directory del progetto
WORKDIR="/home/vbucciero/projects/flood_risk_model"
cd "$WORKDIR"

# Cartelle log/outputs sicure
mkdir -p logs outputs

# Allinea i thread al numero di CPU richieste
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Percorso dell'ambiente (dai tuoi log risulta questo)
ENV_PATH="$HOME/.local/share/mamba/envs/flood"

echo "=== Job info ==="
echo "Workdir: $WORKDIR"
echo "Env path: $ENV_PATH"
echo "CPUs: $OMP_NUM_THREADS"
echo "Start: $(date)"
echo "================"

# Esegui con micromamba (preferito) oppure fallback su conda/mamba
if command -v micromamba >/dev/null 2>&1; then
  micromamba run -p "$ENV_PATH" python main_run_1h.py
else
  # fallback: prova con conda/mamba
  if command -v conda >/dev/null 2>&1; then
    # attiva base e poi env
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python main_run_1h.py
  elif command -v mamba >/dev/null 2>&1; then
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python main_run_1h.py
  else
    echo "Né micromamba né (ana)mamba/conda trovati nel PATH." >&2
    exit 1
  fi
fi

echo "Done: $(date)"
