import rasterio
import numpy as np
from rasterio.enums import Resampling

# Fattore di riduzione (es. 10 significa 1/10 della risoluzione originale)
REDUCTION_FACTOR = 10

def reduce_raster(input_path, output_path, reduction_factor):
    with rasterio.open(input_path) as src:
        # Calcola le nuove dimensioni
        new_width = src.width // reduction_factor
        new_height = src.height // reduction_factor
        
        print("Leggendo il file originale...")
        # Leggi i dati con resampling
        data = src.read(
            1,  # prima banda
            out_shape=(new_height, new_width),
            resampling=Resampling.mode  # usa la moda per preservare i valori D8
        )
        
        print("Preparando il nuovo file...")
        # Crea il nuovo file con metadati aggiornati
        transform = rasterio.Affine(
            src.transform.a * reduction_factor,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e * reduction_factor,
            src.transform.f
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })
        
        print("Salvando il file ridotto...")
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data, 1)
            
        print(f"\nStatistiche:")
        print(f"File originale: {src.width}x{src.height} pixels")
        print(f"File ridotto: {new_width}x{new_height} pixels")
        print(f"Fattore di riduzione: {reduction_factor}x")
        print(f"File salvato come: {output_path}")
        
        # Calcola la riduzione in dimensione del file
        import os
        original_size = os.path.getsize(input_path) / (1024*1024)  # MB
        reduced_size = os.path.getsize(output_path) / (1024*1024)  # MB
        print(f"\nDimensione file originale: {original_size:.1f} MB")
        print(f"Dimensione file ridotto: {reduced_size:.1f} MB")
        print(f"Rapporto di compressione: {original_size/reduced_size:.1f}x")

if __name__ == "__main__":
    input_file = "data/D8_output_italy.tiff"
    output_file = "data/D8_output_italy_reduced.tiff"
    reduce_raster(input_file, output_file, REDUCTION_FACTOR)