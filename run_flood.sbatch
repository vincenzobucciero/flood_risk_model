#!/bin/bash --login
#SBATCH --job-name=runoff_italy
#SBATCH --output=runoff_italy-%j.out
#SBATCH --error=runoff_italy-%j.err

#SBATCH --nodes=1                 # run on a single compute node
#SBATCH --ntasks=1                # number of MPI tasks (processes)
#SBATCH --cpus-per-task=8         # number of CPU cores to use per task
#SBATCH --mem=32G                # total memory requested for the job
#SBATCH --time=24:00:00           # wall‑clock time limit (24 hours)
#SBATCH --partition=gpu       # queue/partition name – adjust for your cluster

# If your environment depends on other modules (e.g. GDAL, OpenMPI), load them here.
# module load GDAL
# module load OpenMPI

# Activate your micromamba environment.  If you are using a Python virtualenv instead,
# comment out these two lines and replace them with a `source /path/to/venv/bin/activate`.
eval "$(micromamba shell hook --shell bash)"
micromamba activate flood

# Navigate to the project directory
cd /home/vbucciero/projects/flood_risk_model

# (Optional) export CLIENT_ID and CLIENT_SECRET if they are not read from a .env file
# export CLIENT_ID=<your-client-id>
# export CLIENT_SECRET=<your-client-secret>

# Run the main Python script.  For a single‑task job, you can call python directly.
# If you need to run with multiple MPI ranks (ntasks > 1), replace python with srun python.
python main.py