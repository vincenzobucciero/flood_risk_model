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
    print(colored("Loading DEM and creating sea mask...", "yellow"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # Verifica esistenza del D8 pre-calcolato
    print(colored("Checking D8 flow direction file...", "yellow"))
    if os.path.exists(config.D8_FILEPATH):
        print(colored(f"Using existing D8 file: {config.D8_FILEPATH}", "green"))
    else:
        raise FileNotFoundError(f"D8 file not found at {config.D8_FILEPATH}. Run the full workflow first to generate it.")

    # Align CN map
    print(colored("Processing Curve Number map...", "yellow"))
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