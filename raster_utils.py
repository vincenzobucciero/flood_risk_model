import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_raster(file_path, title): 
    """
        Carica un file raster e lo visualizza 
        
        args: 
            file_path(string): percorso del file raster
            title(string): titolo della visualizzazione
            
        returns: dati raster e profilo
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        print(f"{title} Metadata:\n", profile)
        plt.figure(figsize=(8,6))
        plt.title(title)
        plt.imshow(data, cmap='terrain' if 'dem' in file_path.lower() else 'Blues')
        plt.colorbar(label='Altitudine (m)' if 'dem' in file_path.lower() else 'Intensità di pioggia (mm/h)')
        plt.show()
        return data, profile
    
def reproject_raster(input_path, reference_profile, output_path):
    """
        Ricalibra raster per la risoluzione
        
        args:
            input_path(string): percorso del raster da ricalibrare
            reference_profile: profilo di riferimento del dem
            output_path(string): percorso del raster ricalibrato
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, reference_profile['crs'], reference_profile['width'], reference_profile['height'], *src.bounds
        )
        
        profile = src.profile.copy()
        profile.update({
            'crs': reference_profile['crs'],
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = dst.transform,
                    dst_crs = reference_profile['crs'],
                    resampling = Resampling.bilinear
                )
        print(f"Raster ricalibrato salvato in {output_path}")