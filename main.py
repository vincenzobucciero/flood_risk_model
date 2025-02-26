from raster_utils import *
from hydrology import *

def main():
    print("Inizio elaborazione dei dati . . . ")
    
    # File paths
    dem_filepath = "data/campania_dem_combined.tiff"
    radar_filepath = "data/rdr0_d01_20250215Z1030_VMI.tiff"
    radar_campania = "data/cropped.tif"
    reproject_raster_filepath = "data/reprojected_radar.tiff"
    cn_map_filepath = "data/cn_map.tif"
    
    # Caricamento DEM
    print("Caricamento DEM . . .")
    dem_data, dem_profile = load_and_plot_raster(dem_filepath, "DEM - Modello Digitale Elevazione")
        
    # Crop del radar
    print("Crop del radar . . .")
    crop_tiff_to_campania(radar_filepath, radar_campania)
    
    # Caricamento dati radar
    print("Caricamento dati radar . . .")
    radar_data, radar_profile = load_and_plot_raster(radar_campania, "Radar - Intensità di pioggia")
    
    # Ricalibrazione del raster radar
    print("Ricalibrazione del raster radar . . .")
    align_radar_to_dem(radar_campania, dem_filepath, reproject_raster_filepath)
    
    # Caricamento radar ricalibrato
    print("Caricamento radar ricalibrato . . .")
    reprojected_rainfall_data, _ = load_and_plot_raster(reproject_raster_filepath, "Radar - Ricalibrato su DEM")
    
    # Riallineamento della mappa CN al DEM (se necessario)
    aligned_cn_filepath = "aligned_cn.tif"
    align_radar_to_dem(cn_map_filepath, dem_filepath, aligned_cn_filepath)
    # Caricamento mappa Curve Number riallineata
    print("Caricamento mappa Curve Number riallineata . . .")
    cn_map, _ = load_and_plot_raster(aligned_cn_filepath, "Mappa Curve Number Riallineata")
    
    # Calcolo del deflusso superficiale (Runoff)
    print("Calcolo del deflusso superficiale . . .")
    runoff = compute_runoff(reprojected_rainfall_data, cn_map)
    
    # Visualizzazione del deflusso superficiale
    plt.figure(figsize=(8, 6))
    plt.title("Deflusso Superficiale (Runoff)")
    plt.imshow(runoff, cmap='Reds')
    plt.colorbar(label='Runoff (mm)')
    plt.show()

if __name__ == "__main__":
    main()