import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# Carica il file NetCDF
print("Caricamento del file NetCDF...")
ds = xr.open_dataset('outputs/runoff_single.nc')

# Stampa informazioni sul dataset
print("\nInformazioni sul dataset:")
print(ds.info())

# Stampa le coordinate
print("\nCoordinate:")
print("Latitudine:", ds.latitude.values.min(), "a", ds.latitude.values.max())
print("Longitudine:", ds.longitude.values.min(), "a", ds.longitude.values.max())

# Ottieni i dati di runoff
runoff_data = ds.runoff.values

# Stampa statistiche di base
print(f"\nStatistiche del runoff:")
print(f"Shape originale: {runoff_data.shape}")
print(f"Min: {np.nanmin(runoff_data):.2f}")
print(f"Max: {np.nanmax(runoff_data):.2f}")
print(f"Media: {np.nanmean(runoff_data):.2f}")
print(f"Mediana: {np.nanmedian(runoff_data):.2f}")

# Downsampling dei dati per la visualizzazione
scale_factor = 0.1  # riduce la dimensione al 10%
new_shape = (int(runoff_data.shape[0] * scale_factor), 
             int(runoff_data.shape[1] * scale_factor))
print(f"Shape dopo il downsampling: {new_shape}")

runoff_downsampled = zoom(runoff_data, scale_factor, order=1)

# Crea una figura con due subplot
plt.figure(figsize=(15, 6))

# Plot dei valori di runoff (con scala logaritmica)
plt.subplot(121)
plt.imshow(np.log1p(runoff_downsampled), cmap='viridis')
plt.colorbar(label='log(1+runoff)')
plt.title('Runoff (log scale)')

# Plot dei valori critici
plt.subplot(122)
threshold = np.nanpercentile(runoff_downsampled, 95)
critical_data = np.where(runoff_downsampled > threshold, runoff_downsampled, np.nan)
plt.imshow(critical_data, cmap='Reds')
plt.colorbar(label='runoff (mm)')
plt.title('Critical Runoff Areas (>95th percentile)')

plt.tight_layout()
plt.savefig('outputs/runoff_single_preview.png', dpi=300, bbox_inches='tight')
print("\nPreview salvata in outputs/runoff_single_preview.png")

# Chiudi il dataset
ds.close()