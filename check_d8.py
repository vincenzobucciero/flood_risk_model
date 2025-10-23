import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Apri il file D8
with rasterio.open('data/D8_output_italy.tiff') as d8:
    # Stampa i metadati
    print("=== Metadati del file D8 ===")
    print(f"Dimensioni: {d8.width}x{d8.height} pixel")
    print(f"Sistema di riferimento: {d8.crs}")
    print(f"Transform: {d8.transform}")
    print(f"Tipo di dati: {d8.dtypes}")
    print(f"Numero di bande: {d8.count}")
    
    # Leggi un'area più piccola dal centro del raster
    window = rasterio.windows.Window(
        d8.width // 4,  # start col
        d8.height // 4,  # start row
        d8.width // 2,  # width
        d8.height // 2   # height
    )
    sample = d8.read(1, window=window)
    
    # Verifica range dei valori (dovrebbero essere 1,2,4,8,16,32,64,128 per D8)
    unique_values = np.unique(sample)
    print("\n=== Valori unici trovati (dovrebbero essere potenze di 2 per D8) ===")
    print(sorted(unique_values))
    
    # Riduci ulteriormente per la visualizzazione
    sample_small = sample[::20, ::20]
    
    # Crea una visualizzazione
    plt.figure(figsize=(12, 8))
    plt.imshow(sample_small, cmap='viridis')
    plt.colorbar(label='Direzione di flusso D8')
    plt.title('Anteprima direzioni di flusso D8 (area centrale)')
    plt.savefig('d8_preview.png', dpi=300, bbox_inches='tight')
    print("\nAnteprima salvata come 'd8_preview.png'")