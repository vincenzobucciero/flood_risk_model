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

