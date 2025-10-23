#!/bin/bash --login
## SBATCH script to process prediction TIFFs on the 'rainbow' HPC using an array job.
## Each array task will process one TIFF and write outputs/<tiff_stem>.nc
#SBATCH --job-name=runoff_pred
#SBATCH --output=runoff_rainbow-%A_%a.out
#SBATCH --error=runoff_rainbow-%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=rainbow

### NOTE: Adjust --mem, --cpus-per-task and --time according to the node specs and
### processing profile. For full-Italy alignment the memory requirement is higher.

set -eo pipefail
# Protect against conda/micromamba deactivate hooks referencing unset variables
export CONDA_BACKUP_CXX=${CONDA_BACKUP_CXX:-}

echo "Starting SLURM task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

eval "$(~/projects/flood_risk_model/bin/micromamba shell hook --shell bash)"
micromamba activate flood

cd /home/vbucciero/projects/flood_risk_model

# Build list of files and pick the one matching the array index (1-based)
PRED_DIR="${PRED_DIR:-/storage/external_01/hiwefi/data/rdr0_val_previews/epoch_000/predictions}"
FILES=("$(ls -1 $PRED_DIR/*.tiff)")

if [ -z "${FILES[*]}" ]; then
  echo "No TIFFs found in $PRED_DIR"; exit 1
fi

# Convert to bash array properly
mapfile -t FILE_LIST < <(ls -1 "$PRED_DIR"/*.tiff)
IDX=$((SLURM_ARRAY_TASK_ID-1))
FILE=${FILE_LIST[$IDX]}

echo "SLURM task processing: $FILE"

# Run the per-file processing; the Python script will save outputs/<basename>.nc
python - <<PY
from hydrology import process_radar_file, save_as_netcdf
from raster_utils import latlon_load_and_plot_land_cover, sea_mask, align_radar_to_dem
import config, os

# ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# ensure aligned CN exists; if not, create it
if not os.path.exists(config.ALIGNED_CN_FILEPATH):
  print('Aligned CN not found; creating', config.ALIGNED_CN_FILEPATH)
  align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)

cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)
mask = sea_mask(config.DEM_FILEPATH)

file = r"$FILE"
print('Processing', file)
runoff = process_radar_file(file, cn_map, mask, config.DEM_FILEPATH)
if runoff is None:
  raise SystemExit('Processing failed for ' + file)
out_path = os.path.join('outputs', os.path.splitext(os.path.basename(file))[0] + '.nc')
save_as_netcdf(runoff, out_path, config.DEM_FILEPATH)
print('Saved', out_path)
PY

echo "Task completed"
