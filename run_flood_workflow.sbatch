#!/bin/bash
#SBATCH --job-name=flood_risk        # Nome del job
#SBATCH --output=logs/flood_%j.out   # File di output (%j è l'ID del job)
#SBATCH --error=logs/flood_%j.err    # File di error (%j è l'ID del job)
#SBATCH --time=24:00:00             # Tempo massimo di esecuzione (24 ore)
#SBATCH --nodes=1                    # Numero di nodi richiesti
#SBATCH --ntasks-per-node=1         # Processi per nodo
#SBATCH --cpus-per-task=4           # CPU per processo
#SBATCH --mem=16G                   # Memoria richiesta
#SBATCH --partition=gpu            # Partizione da utilizzare

# Crea directory per i log se non esiste
mkdir -p logs

eval "$(~/projects/flood_risk_model/bin/micromamba shell hook --shell bash)"
micromamba activate flood

# Crea directory outputs se non esiste
mkdir -p outputs

# Imposta il numero di threads OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Stampa informazioni sull'ambiente
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Esegui il workflow completo per una singola previsione
echo "Starting main workflow..."
srun python -u main.py  # -u forza l'output unbuffered

# Se il workflow principale è completato con successo, genera la versione ridotta del NetCDF
if [ $? -eq 0 ]; then
    echo "Main workflow completed successfully. Creating reduced NetCDF..."
    python reduce_netcdf.py
    
    # Genera visualizzazione
    echo "Generating visualization..."
    python visualize_runoff.py
fi

echo "Finished at: $(date)"

# Mostra utilizzo risorse
scontrol show job $SLURM_JOB_ID