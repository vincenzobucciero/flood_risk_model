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
    dem_data, dem_profile = load_and_plot_dem(dem_filepath)
        
    # Crop del radar
    print("Crop del radar . . .")
    crop_tiff_to_campania(radar_filepath, radar_campania)
    
    # Caricamento dati radar
    print("Caricamento dati radar . . .")
    radar_data, radar_profile = load_and_plot_rainfall(radar_campania)
    
    # Ricalibrazione del raster radar
    print("Ricalibrazione del raster radar . . .")
    align_radar_to_dem(radar_campania, dem_filepath, reproject_raster_filepath)
    
    # Caricamento radar ricalibrato
    print("Caricamento radar ricalibrato . . .")
    reprojected_rainfall_data, _ = load_and_plot_rainfall(reproject_raster_filepath, "Radar - Ricalibrato su DEM")
    
    # Riallineamento della mappa CN al DEM (se necessario)
    aligned_cn_filepath = "aligned_cn.tif"
    align_radar_to_dem(cn_map_filepath, dem_filepath, aligned_cn_filepath)
    # Caricamento mappa Curve Number riallineata
    print("Caricamento mappa Curve Number riallineata . . .")
    cn_map, _ = load_and_plot_land_cover(aligned_cn_filepath)
    
    # Calcolo del deflusso superficiale (Runoff)
    print("Calcolo del deflusso superficiale . . .")
    runoff = compute_runoff(reprojected_rainfall_data, cn_map)
    plot_runoff_on_land_cover(cn_map, runoff)
    
    print("Calcolo del D8 . . .")
    calculate_flow_direction_parallel(dem_filepath, "D8_output.tiff")
    
    flood_risk_map = calculate_flood_risk("D8_output.tiff", runoff)
    visualize_flood_risk(flood_risk_map)

if __name__ == "__main__":
    main()