#!/bin/bash
#SBATCH --job-name=runoff_italy
#SBATCH --nodes=1
#SBATCH --ntasks=1           # un solo processo
#SBATCH --cpus-per-task=10    # usa 10 core; puoi aumentare fino a 20
#SBATCH --mem=88G            # lascia un margine al sistema
#SBATCH --time=48:00:00      # adattalo alle tue esigenze
#SBATCH --partition=h100gpu  # o la coda del tuo cluster
#SBATCH --output=runoff_italy-%j.out
#SBATCH --error=runoff_italy-%j.err

# carica micromamba (o conda) dall’ambiente
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate flood

# vai nella directory del progetto
cd /home/v.bucciero/projects/hiwefi/flood_risk_model

# esegui lo script
python main.py
