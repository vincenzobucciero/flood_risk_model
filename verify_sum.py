import xarray as xr
import numpy as np
import glob
import os

# Percorso agli output
out_dir = "/storage/external_01/hiwefi/flood_outputs"

# Carica i sei file singoli
files = sorted(glob.glob(os.path.join(out_dir, "floodrisk_20250725Z*.nc")))
datasets = [xr.open_dataset(f) for f in files]

# Somma dei 6 floodrisk singoli
sum_6 = sum(ds["runoff"] for ds in datasets)  # se la variabile si chiama diversamente, ad es. 'floodrisk', cambia qui

# Carica il floodrisk 1h
flood_1h = xr.open_dataset(os.path.join(out_dir, "floodrisk_1h_20250725Z1800.nc"))["runoff"]

# Confronto numerico
diff = np.abs(sum_6 - flood_1h)
print("Differenza media:", float(diff.mean()))
print("Differenza massima:", float(diff.max()))
