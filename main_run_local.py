import os
from raster_utils import *
from hydrology import *
from termcolor import colored
import config

def main():
    print(colored("Test run: processing a single prediction TIFF (local)", "cyan"))
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Load DEM and mask
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # Align CN map (assumes ALIGNED_CN_FILEPATH already exists)
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # Run runoff calculation on the single prediction file
    print(colored(f"Using prediction file: {config.PREDICTION_FILE}", "yellow"))
    runoff = process_radar_file(
        config.PREDICTION_FILE,
        cn_map,
        MASK,
        config.DEM_FILEPATH
    )
    
    # Save output as NetCDF
    print(colored("Saving output as NetCDF...", "yellow"))
    save_as_netcdf(runoff, "outputs/runoff_single.nc", config.DEM_FILEPATH)

    print(colored("Runoff computation finished (local test)", "green"))

if __name__ == '__main__':
    main()