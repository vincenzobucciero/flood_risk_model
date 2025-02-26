import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os

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
    
# questa sotto non viene utilizzata
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
        
def align_radar_to_dem(radar_tiff, dem_tiff, output_radar_tiff):
    """
    Riproietta e riallinea un file radar al DEM senza distorsioni.

    :param radar_tiff: Percorso del file TIFF del radar.
    :param dem_tiff: Percorso del file TIFF del DEM.
    :param output_radar_tiff: Percorso del file TIFF risultante dopo l'allineamento.
    """
    with rasterio.open(dem_tiff) as dem:
        dem_crs = dem.crs
        dem_transform = dem.transform
        dem_width = dem.width
        dem_height = dem.height
        dem_bounds = dem.bounds
        dem_resolution_x = dem_transform[0]  
        dem_resolution_y = -dem_transform[4]  

    with rasterio.open(radar_tiff) as radar:
        radar_crs = radar.crs

        print(f"CRS DEM: {dem_crs}, CRS Radar: {radar_crs}")
        print(f"DEM Risoluzione: {dem_resolution_x}, {dem_resolution_y}")

        if radar_crs != dem_crs:
            print("⚠️ ATTENZIONE: Il CRS del radar è diverso da quello del DEM. Viene effettuata la conversione.")

    options = gdal.WarpOptions(
        format = "GTiff",
        dstSRS = str(dem_crs),
        xRes = dem_resolution_x,  
        yRes = dem_resolution_y,  
        width = dem_width,  
        height = dem_height, 
        outputBounds = [dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top],
        resampleAlg = gdal.GRA_Cubic  
    )

    gdal.Warp(output_radar_tiff, radar_tiff, options=options)
    print(f"✅ File radar riallineato salvato in: {output_radar_tiff}") 
    
def crop_tiff_to_campania(input_tiff, output_tiff):
    """
    Esegue il cropping di un file TIFF sull'area della Campania.

    :param input_tiff: Percorso del file TIFF da cui eseguire il crop.
    :param output_tiff: Percorso del file TIFF risultante dopo il crop.
    """
    # Definisci il bounding box per la Campania
    campania_bbox = {
        "min_x": 13.7,  # Longitudine Ovest
        "max_x": 15.8,  # Longitudine Est
        "min_y": 39.9,  # Latitudine Sud
        "max_y": 41.5   # Latitudine Nord
    }

    # Verifica che il file di input esista
    if not os.path.exists(input_tiff):
        raise FileNotFoundError(f"Il file {input_tiff} non esiste.")

    try:
        # Usa gdal.Translate per eseguire il cropping
        gdal.Translate(
            output_tiff,
            input_tiff,
            projWin=[campania_bbox["min_x"], campania_bbox["max_y"], campania_bbox["max_x"], campania_bbox["min_y"]]
        )
        print(f"Cropping completato. File salvato in: {output_tiff}")
    except Exception as e:
        print(f"Errore durante il cropping: {e}")

# da cancellare questa funzione sotto perchè non utilizzata.
def process_tiff(input_tiff, bbox=None, tile_width=0.01, tile_height=0.01, output_dir="data/"):
    """
    Processa un file TIFF: lo croppa in base a un bbox e lo suddivide in tile.

    :param input_tiff: Percorso del file TIFF di input.
    :param bbox: Bounding box specificato come {"min_x", "max_x", "min_y", "max_y"}.
                 Se None, verrà usato un bbox di default.
    :param tile_width: Larghezza del tile in gradi decimali (default: 0.01).
    :param tile_height: Altezza del tile in gradi decimali (default: 0.01).
    :param output_dir: Cartella di output per i tile generati.
    """
    # Definisci un bbox di default se non viene fornito
    if bbox is None:
        bbox = { # solofra
            "min_x": 14.8101704568,
            "max_x": 14.880832918,
            "min_y": 40.7967829884,
            "max_y": 40.8585629063
        }

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Calcola il numero di tile necessari
    num_tiles_x = int(np.ceil((bbox["max_x"] - bbox["min_x"]) / tile_width))
    num_tiles_y = int(np.ceil((bbox["max_y"] - bbox["min_y"]) / tile_height))

    # Step 1: Crop the TIFF based on the bbox
    cropped_file = os.path.join(output_dir, "cropped.tif")
    try:
        gdal.Translate(
            cropped_file,
            input_tiff,
            projWin=[bbox["min_x"], bbox["max_y"], bbox["max_x"], bbox["min_y"]]
        )
        print(f"Cropping completato. File salvato in: {cropped_file}")
    except Exception as e:
        print(f"Errore durante il cropping: {e}")
        return

    # Step 2: Generate tiles from the cropped TIFF
    for tile_x in range(num_tiles_x):
        for tile_y in range(num_tiles_y):
            tile_min_x = bbox["min_x"] + tile_x * tile_width
            tile_max_x = min(tile_min_x + tile_width, bbox["max_x"])
            tile_min_y = bbox["min_y"] + tile_y * tile_height
            tile_max_y = min(tile_min_y + tile_height, bbox["max_y"])

            tile_filename = os.path.join(output_dir, f"tiles/tile_{tile_x}_{tile_y}.tiff")
            try:
                gdal.Translate(
                    tile_filename,
                    cropped_file,
                    projWin=[tile_min_x, tile_max_y, tile_max_x, tile_min_y]
                )
                print(f"Tile generato: {tile_filename}")
            except Exception as e:
                print(f"Errore durante la generazione del tile ({tile_x}, {tile_y}): {e}")

    print("Processo completato. Tutti i tile sono stati generati con successo.")
    
def visualize_combined(dem_data, radar_original_data, radar_reprojected_data):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot DEM
    im0 = axes[0].imshow(dem_data, cmap='terrain')
    axes[0].set_title("DEM", fontsize=14)
    axes[0].axis('off')  # Nasconde gli assi
    fig.colorbar(im0, ax=axes[0], label='Altitudine (m)', fraction=0.046, pad=0.04)
    
    # Plot Radar originale
    im1 = axes[1].imshow(radar_original_data, cmap='Blues', vmin=np.min(radar_reprojected_data), vmax=np.max(radar_reprojected_data))
    axes[1].set_title("Radar Originale", fontsize=14)
    axes[1].axis('off')  # Nasconde gli assi
    fig.colorbar(im1, ax=axes[1], label='Intensità di pioggia (mm/h)', fraction=0.046, pad=0.04)
    
    # Plot Radar ricalibrato
    im2 = axes[2].imshow(radar_reprojected_data, cmap='Blues', vmin=np.min(radar_reprojected_data), vmax=np.max(radar_reprojected_data))
    axes[2].set_title("Radar Ricalibrato", fontsize=14)
    axes[2].axis('off')  # Nasconde gli assi
    fig.colorbar(im2, ax=axes[2], label='Intensità di pioggia (mm/h)', fraction=0.046, pad=0.04)
    
    plt.tight_layout()  
    plt.show()
    

def plot_flood_risk_map(dem_data, runoff_map, flow_accumulation, flood_risk):
    plt.figure(figsize=(10, 8))
    plt.title("Mappa delle Zone a Rischio di Alluvione")
    
    # Sfondo DEM
    plt.imshow(dem_data, cmap="gray", alpha=0.5)
    
    # Sovrapposizione runoff (deflusso superficiale)
    plt.imshow(runoff_map, cmap="Blues", alpha=0.4)

    # Sovrapposizione accumulo del flusso
    plt.imshow(flow_accumulation, cmap="Greens", alpha=0.3)

    # Evidenziazione zone ad alto rischio di alluvione
    plt.imshow(flood_risk, cmap="Reds", alpha=0.6)
    
    plt.colorbar(label="Indice di rischio")
    plt.show()