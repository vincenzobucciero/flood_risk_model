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
![NOTE] If ```richdem``` fails to build (missing Python.h), you need Python dev headers:
```bash
sudo apt install python3-dev build-essential
```
