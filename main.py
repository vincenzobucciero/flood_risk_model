import os
from raster_utils import *
from hydrology import *
from termcolor import colored
import config

def main():
    print(colored("Starting flood risk data processing...", "cyan"))
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Load DEM and mask
    print(colored("Loading DEM data...", "green"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # Align CN map if needed
    print(colored("Processing Curve Number map...", "green"))
    if not os.path.exists(config.ALIGNED_CN_FILEPATH):
        align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # Run runoff calculation on the prediction file
    print(colored(f"Processing radar prediction: {config.PREDICTION_FILE}", "yellow"))
    runoff = process_radar_file(
        config.PREDICTION_FILE,
        cn_map,
        MASK,
        config.DEM_FILEPATH
    )
    
    # Save output as NetCDF
    print(colored("Saving output as NetCDF...", "yellow"))
    save_as_netcdf(runoff, "outputs/runoff_single.nc", config.DEM_FILEPATH)

    # Calculate D8 flow direction
    print(colored("Calculating D8 flow direction...", "blue"))
    d8_initialize(config.DEM_FILEPATH, config.D8_FILEPATH)

    # Compute and visualize flood risk
    print(colored("Computing flood risk map...", "blue"))
    flood_risk_map = compute_flood_risk(config.D8_FILEPATH, runoff)
    visualize_flood_risk_with_legend(flood_risk_map, dem_data)
    
    print(colored("Processing completed successfully!", "cyan"))

if __name__ == "__main__":
    main()


from raster_utils import *
from hydrology import *
from termcolor import colored
import script.download
import config

def main():
    print(colored("Avvio del processo di elaborazione dei dati...", "cyan"))

    # Scarica e prepara DEM e GLC per l’Italia
    script.download.main()

    # Caricamento DEM e creazione della maschera mare
    print(colored("Caricamento del DEM in corso...", "green"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # Riallineamento della mappa Curve Number al DEM italiano
    print(colored("Riallineamento della mappa Curve Number al DEM...", "green"))
    align_radar_to_dem(
        config.CN_MAP_FILEPATH,
        config.DEM_FILEPATH,
        config.ALIGNED_CN_FILEPATH
    )

    # Caricamento della CN map riallineata
    print(colored("Caricamento della mappa Curve Number riallineata...", "yellow"))
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # Calcolo del deflusso superficiale (runoff) usando le predizioni del tuo modello
    print(colored("Calcolo del deflusso superficiale...", "blue"))
    # Il risultato verrà salvato come "runoff_italy.nc" (NetCDF)
    runoff = calculate_accumulated_runoff(
        config.PREDICTION_DIR,
        cn_map,
        MASK,
        config.DEM_FILEPATH,
        "runoff_italy",
        output_format="netcdf"
    )
    plot_territory_boundaries(dem_data, runoff)

    # Calcolo della direzione del flusso (D8) e del rischio di alluvione
    print(colored("Calcolo della direzione del flusso D8...", "blue"))
    d8_initialize(config.DEM_FILEPATH, config.D8_FILEPATH)

    flood_risk_map = compute_flood_risk(config.D8_FILEPATH, runoff)
    visualize_flood_risk_with_legend(flood_risk_map, dem_data)

    print(colored("Processo completato con successo!", "cyan"))

if __name__ == "__main__":
    main()
