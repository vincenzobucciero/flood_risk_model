#!/bin/bash
#SBATCH -J flood_dyn          # Nome del job
#SBATCH -p gpu                # Partizione; sostituire se necessario (es. cpu, standard, long)
#SBATCH -c 8                  # Numero di CPU per task
#SBATCH --mem=32G             # Memoria RAM
#SBATCH -t 04:00:00           # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH -o logs/flood_dyn-%j.out  # File di log stdout
#SBATCH -e logs/flood_dyn-%j.err  # File di log stderr

# Script batch per eseguire la pipeline di flooding dinamico su HPC Rainbow.
# Salva anche un log live nel file logs/flood_dyn-<jobid>.live.log che può
# essere seguito con `tail -f` per monitorare l'avanzamento.

set -euo pipefail

# Directory di lavoro e ambiente
WORKDIR="/home/vbucciero/projects/flood_risk_model"
ENV_PATH="$HOME/.local/share/mamba/envs/flood"

cd "$WORKDIR"

# Crea directory logs se non esiste
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs: $OMP_NUM_THREADS"
echo "Working directory: $WORKDIR"
echo "Environment: $ENV_PATH"
echo "Start time: $(date)"
echo "================="

# Esecuzione del programma
if command -v micromamba >/dev/null 2>&1; then
  echo "[INFO] Uso micromamba per eseguire l'ambiente." | tee >(cat >&2)
  micromamba run -p "$ENV_PATH" python run_test.py 2>&1 |
    tee logs/flood_dyn-${SLURM_JOB_ID}.live.log
else
  echo "[INFO] Uso conda/mamba per attivare l'ambiente." | tee >(cat >&2)
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python run_test.py 2>&1 |
      tee logs/flood_dyn-${SLURM_JOB_ID}.live.log
  elif command -v mamba >/dev/null 2>&1; then
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH" 2>/dev/null || conda activate flood
    python run_test.py 2>&1 |
      tee logs/flood_dyn-${SLURM_JOB_ID}.live.log
  else
    echo "[ERROR] Non è stato trovato micromamba, conda o mamba nel PATH." >&2
    exit 1
  fi
fi

echo "=== Job completed ==="
echo "End time: $(date)"