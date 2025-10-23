from raster_utils import *
from hydrology import *
from termcolor import colored
import config

def main():
    print(colored("Test run: processing a single prediction TIFF (local)", "cyan"))

    # Load DEM and mask
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # Align CN map (assumes ALIGNED_CN_FILEPATH already exists)
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # Run runoff calculation on the single prediction file
    print(colored(f"Using prediction file: {config.PREDICTION_FILE}", "yellow"))
    runoff = calculate_accumulated_runoff(
        os.path.dirname(config.PREDICTION_FILE),
        cn_map,
        MASK,
        config.DEM_FILEPATH,
        "runoff_test",
        output_format="netcdf"
    )

    print(colored("Runoff computation finished (local test)", "green"))

if __name__ == '__main__':
    main()