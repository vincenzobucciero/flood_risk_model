#!/bin/bash
#SBATCH --job-name=runoff_italy
#SBATCH --ntasks=1           # un solo processo
#SBATCH --cpus-per-task=10    # usa 10 core; puoi aumentare fino a 20
#SBATCH --mem=32G            # lascia un margine al sistema
#SBATCH --time=06:00:00      # adattalo alle tue esigenze
#SBATCH --partition=xhicpu  # o la coda del tuo cluster
#SBATCH --chdir=/home/v.bucciero/projects/hiwefi/flood_risk_model
#SBATCH --output=runoff_italy-%j.out
#SBATCH --error=runoff_italy-%j.err

# carica micromamba (o conda) dall’ambiente
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate flood

set -a
source /home/v.bucciero/projects/hiwefi/flood_risk_model/.env
set +a

# vai nella directory del progetto
cd /home/v.bucciero/projects/hiwefi/flood_risk_model

# disabilita il file locking HDF5 su FS di rete
export HDF5_USE_FILE_LOCKING=FALSE
# (opzionale ma utile) evita cache di chunk HDF5 troppo aggressive
export HDF5_DISABLE_VERSION_CHECK=2

# esegui lo script
python -u main.py
