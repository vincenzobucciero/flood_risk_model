import xarray as xr
import numpy as np

# Percorsi ai file (modifica se necessario)
flood_1300_path = "/storage/external_01/hiwefi/flood_outputs/floodrisk_20250725Z1300.nc"
flood_1310_path = "/storage/external_01/hiwefi/flood_outputs/floodrisk_20250725Z1310.nc"

# Leggi i due file (variabile: runoff)
flood_1300 = xr.open_dataset(flood_1300_path)["runoff"].values
flood_1310 = xr.open_dataset(flood_1310_path)["runoff"].values

# Leggi anche il runoff istantaneo del passo 13:10
runoff_1310_path = "/storage/external_01/hiwefi/flood_outputs/runoff_20250725Z1310.nc"
runoff_1310 = xr.open_dataset(runoff_1310_path)["runoff"].values

# Somma flood precedente + runoff istantaneo (clip 0–1)
flood_predicted = np.clip(flood_1300 + runoff_1310, 0, 1)

# Differenza rispetto al flood dinamico del file successivo
diff = flood_1310 - flood_predicted
print("Differenza media:", np.nanmean(diff))
print("Differenza massima:", np.nanmax(np.abs(diff)))

# Mostra quanto del dominio coincide
eq = np.isclose(flood_1310, flood_predicted, atol=1e-5)
print(f"Celle identiche (entro tolleranza 1e-5): {np.sum(eq)} / {eq.size}")
