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
    d8_filepath = "D8_output.tiff"
    
    # Caricamento DEM
    print("Caricamento DEM . . .")
    dem_data, dem_profile = latlon_load_and_plot_dem(dem_filepath)
    #plot_raster_with_latlon(dem_filepath)
    MASK = sea_mask(dem_filepath)
        
    # Crop del radar
    print("Crop del radar . . .")
    crop_tiff_to_campania(radar_filepath, radar_campania)
    
    # Caricamento dati radar
    print("Caricamento dati radar . . .")
    radar_data, radar_profile = latlon_load_and_plot_rainfall(radar_campania)
    # plot_raster_with_latlon(radar_campania)
    
    # Ricalibrazione del raster radar
    print("Ricalibrazione del raster radar . . .")
    align_radar_to_dem(radar_campania, dem_filepath, reproject_raster_filepath)
    
    # Caricamento radar ricalibrato
    print("Caricamento radar ricalibrato . . .")
    reprojected_rainfall_data, _ = latlon_load_and_plot_rainfall(reproject_raster_filepath, "Radar - Ricalibrato su DEM")
    # plot_raster_with_latlon(reproject_raster_filepath)
    
    # Riallineamento della mappa CN al DEM (se necessario)
    aligned_cn_filepath = "aligned_cn.tif"
    align_radar_to_dem(cn_map_filepath, dem_filepath, aligned_cn_filepath)
    # Caricamento mappa Curve Number riallineata
    print("Caricamento mappa Curve Number riallineata . . .")
    cn_map, _ = latlon_load_and_plot_land_cover(aligned_cn_filepath)
    
    # Calcolo del deflusso superficiale (Runoff)
    print("Calcolo del deflusso superficiale . . .")
    runoff = compute_runoff(reprojected_rainfall_data, cn_map, MASK)
    # plot_runoff_on_land_cover(cn_map, runoff)
    plot_territory_boundaries(dem_data, runoff)
    
    print("Calcolo del D8 . . .")
    calculate_flow_direction_parallel(dem_filepath, d8_filepath, MASK)
    
    flood_risk_map = compute_flood_risk(d8_filepath, runoff)
    # visualize_flood_risk_with_legend(flood_risk_map)
    visualize_flood_risk_with_legend(flood_risk_map, dem_data)
    
    print("Processo completato.")

if __name__ == "__main__":
    main()