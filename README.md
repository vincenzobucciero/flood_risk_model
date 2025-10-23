# 🌊 Flood Risk Model

**Flood Risk Model** is a Python-based framework for **hydrological analysis and flood risk assessment**.  
It integrates multiple geospatial datasets — including **Digital Elevation Model (DEM)**, **Land Cover (GLC)**, **Curve Number (CN)**, and **radar precipitation data** — to compute **surface runoff** and generate a **flood risk map** using D8 flow direction algorithms.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Option 1: Micromamba (recommended)](#option-1-micromamba-recommended)
  - [Option 2: Python Virtual Environment](#option-2-python-virtual-environment)
- [Configuration](#configuration)
- [Execution](#execution)
- [Generated Outputs](#generated-outputs)
- [Troubleshooting](#troubleshooting)
- [License & Author](#license--author)

---

## 🧭 Overview

This project automates a full end-to-end hydrological workflow:

1. **Download and merge DEM tiles**
2. **Download and convert GLC (Global Land Cover) data**
3. **Crop and align radar precipitation data**
4. **Align and resample the Curve Number (CN) map**
5. **Compute accumulated surface runoff (NetCDF)**
6. **Generate D8 flow direction maps**
7. **Produce a flood risk map and visualization**

The pipeline leverages **GDAL**, **Rasterio**, **RichDEM**, and **NumPy/SciPy** for raster processing,  
and supports distributed execution via **MPI (mpi4py)**.

---

## 📂 Project Structure

| Path | Description |
|------|-------------|
| **`bin/`** | Contains micromamba binary for environment management |
| **`data/`** | Input and intermediate data files |
| ├─ `italy_dem_combined.tiff` | Combined DEM for Italy |
| ├─ `glc_italy.tif` | Land cover data for Italy |
| ├─ `aligned_cn_italy.tif` | Aligned Curve Number map |
| **`outputs/`** | Generated output files |
| **`script/`** | Data download and preparation scripts |
| **Main Files** | |
| `config.py` | Configuration and paths |
| `hydrology.py` | Core hydrological computations |
| `raster_utils.py` | Raster processing utilities |
| `main.py` | Main processing script |
| `main_run_local.py` | Local testing script |
| `run_all_predictions.py` | Batch processing script |
| `run_flood_rainbow.sbatch` | SLURM job script |
| `visualize_runoff.py` | Visualization utilities |

---

> **Tip:** The folder hierarchy is automatically created during runtime; ensure `data/` exists before execution.












---

## 💻 System Requirements

| Component | Minimum Requirement |
|------------|--------------------|
| **OS** | Linux (tested on Ubuntu 22.04 / Rainbow cluster) |
| **Python** | 3.10 or later |
| **Memory** | ≥ 8 GB RAM recommended |
| **Dependencies** | GDAL ≥ 3.8, NetCDF, C++ compiler (for `richdem`) |

---

## ⚙️ Installation

### Option 1: Micromamba (recommended, no `sudo` required)

```bash
# Install micromamba (only once)
export MAMBA_ROOT_PREFIX=~/micromamba
curl -Ls https://micro.mamba.pm/install.sh | bash
source ~/.bashrc

# Create environment with all scientific dependencies
micromamba create -n flood -c conda-forge \
  python=3.10 gdal=3.10.* rasterio=1.4.3 richdem=2.3 \
  netcdf4 numpy scipy shapely pandas mpi4py mpich

micromamba activate flood

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Option 2: Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> **Tip:** If ```richdem``` fails to build (missing Python.h), you need Python dev headers:
```bash
sudo apt install python3-dev build-essential
```

## Configuration
Before running, make sure your file paths are correctly defined in ```config.py```:
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DEM_FILEPATH = DATA_DIR / "campania_dem_combined.tiff"
RADAR_FILEPATH = DATA_DIR / "radar.tiff"
RADAR_CAMPANIA = DATA_DIR / "cropped.tif"
CN_MAP_FILEPATH = BASE_DIR / "aligned_cn.tif"
ALIGNED_CN_FILEPATH = BASE_DIR / "aligned_cn.tif"
D8_FILEPATH = DATA_DIR / "d8.tif"
RUNOFF_PATH = DATA_DIR / "runoff.nc"
```

You can also configure custom directories for outputs, logging, and temporary tiles.

## Execution
To start the entire processing pipeline:
```python
python main.py
```
This will automatically:
- Download DEM, GLC, and radar data
- Merge and align all tiles
- Compute the accumulated runoff
- Generate flow direction (D8)
- Produce the flood risk map
Example console output:
```python
Starting flood risk data processing...
DEM tiles downloaded successfully...
GLC data converted to GeoTIFF...
Surface runoff computation in progress...
Process completed successfully!
```
If running in a distributed environment (MPI):
```python
mpirun -n 4 python main.py
```

## 📤 Generated Outputs

| Output File | Type | Description |
|--------------|------|--------------|
| **`data/campania_dem_combined.tiff`** | 🗺️ GeoTIFF | Final merged **Digital Elevation Model (DEM)** of the study area |
| **`data/cropped.tif`** | 🛰️ GeoTIFF | **Radar precipitation** dataset cropped to the Campania region |
| **`aligned_cn.tif`** | 🧭 GeoTIFF | **Curve Number (CN)** map resampled and aligned with the DEM grid |
| **`data/runoff.nc`** | 💧 NetCDF | Computed **accumulated surface runoff** across the DEM surface |
| **`data/d8.tif`** | 🔢 GeoTIFF | **Flow direction map** generated using the **D8 algorithm** |
| **`data/flood_risk.png`** | 🖼️ PNG | **Final flood risk visualization**, combining runoff and flow analysis |

---

> 📦 **Note:** All output files are automatically saved inside the `data/` directory during execution.  
> If running on a remote server (SSH), use `matplotlib.use("Agg")` to save plots instead of displaying them interactively.

---

## Visualization Example
```python
import matplotlib
matplotlib.use("Agg")  # ensures headless plotting on SSH
import matplotlib.pyplot as plt
from rasterio.plot import show

show("data/flood_risk.png", title="Flood Risk Map")
plt.savefig("data/flood_risk_preview.png", dpi=200)
```
> **Tip:** When running on remote SSH (no GUI), always save plots to file instead of displaying interactively.


---

## 🧪 Troubleshooting

Below are common issues you might encounter during setup or execution, along with their recommended fixes.

| Issue | Cause | Solution |
|--------|--------|-----------|
| **`FileNotFoundError: data/test_26mar`** | The radar input directory does not exist or is misconfigured. | Create the folder or update the path in `main.py`:<br>```python<br>runoff = calculate_accumulated_runoff("data", cn_map, MASK, config.DEM_FILEPATH, "runoff")<br>``` |
| **`Python.h: No such file or directory`** | Python development headers are missing (required by `richdem`). | Install Python dev tools:<br>```bash<br>sudo apt install python3-dev build-essential<br>``` |
| **`rasterio` / `GDAL` version mismatch** | Incompatible versions between GDAL and Rasterio. | Reinstall consistent versions:<br>```bash<br>micromamba install -n flood -c conda-forge gdal=3.10.* rasterio=1.4.3<br>``` |
| **`richdem` build fails (no sudo)** | The package requires compilation. | Use the precompiled conda version or replace with:<br>```bash<br>pip install pysheds<br>``` |
| **Plots do not display over SSH** | No GUI backend available in remote environments. | Force Matplotlib to use a non-interactive backend:<br>```python<br>import matplotlib<br>matplotlib.use("Agg")<br>``` |

---

## 🧬 Scientific Context

The **Flood Risk Model** combines hydrological and geomorphological principles to estimate surface runoff and flood susceptibility.  
It integrates:
- The **Curve Number (CN)** method for runoff estimation  
- The **D8 flow direction** algorithm for drainage modeling  
- Spatial datasets from **Copernicus** and **ECMWF** for environmental and climatic context  

This workflow supports reproducible research and HPC deployment for large-scale flood modeling.

---

## � Future Developments

The following enhancements are planned or under consideration:

### Alert System
A comprehensive flood alert system could be implemented with the following features:
- Real-time monitoring of critical runoff thresholds
- Integration with weather forecast data
- Geospatial analysis of risk zones
- Automated notification system (email, SMS, etc.)
- Interactive monitoring dashboard
- Integration with local emergency response systems

The alert system would enhance the model's practical utility for disaster prevention and emergency response.

---

## �📜 License & Author

| Field | Information |
|--------|-------------|
| **Author** | *Vincenzo Bucciero* |
| **Project** | *Flood Risk Model — Rainbow HPC Environment* |
| **License** | MIT License |
| **Contact** | [GitHub Profile](https://github.com/vincenzobucciero) |

> Developed for scientific and academic use.  
> If you use this project in research, please cite it appropriately.

---

## 🏁 Quick Start Summary

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/flood_risk_model.git
cd flood_risk_model

# 2. Create and activate environment
micromamba create -n flood -c conda-forge python=3.10 gdal rasterio richdem
micromamba activate flood

# 3. Install dependencies
pip install -r requirements.txt

# 4. Adjust configuration paths in config.py if needed

# 5. Run the model
python main.py
