from raster_utils import load_and_plot_raster, reproject_raster
from hydrology import compute_hydrology

def main():
    print("Inizio main")
    dem_filepath = "data/dem.tiff"
    radar_filepath = "data/radar.tiff"
    reproject_raster_filepath = "data/reprojected_radar.tiff"
    
    print("Caricamento DEM . . .")
    dem_data, dem_profile = load_and_plot_raster(dem_filepath, "DEM - Modello Digitale Elevazione")
    
    print("Caricamento dati radar . . .")
    radar_data, radar_profile = load_and_plot_raster(radar_filepath, "Radar - Intensità di pioggia")
    
    print("Ricalibrazione del raster radar . . .")
    reproject_raster(radar_filepath, dem_profile, reproject_raster_filepath)
    
    print("Caricamento radar ricalibrato . . .")
    reproject_raster_data, _ = load_and_plot_raster(reproject_raster_filepath, "Radar - Ricalibrato su DEM")
    
    print("Calcolo idrologico ...")
    dem_slope, dem_flow_dir, dem_flow_acc = compute_hydrology(dem_data)
    
if __name__ == "__main__":
    main()