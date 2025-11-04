import glob
import os

'''
# Cartella base (dove si trova questo file)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Percorsi dei file
DEM_FILEPATH = os.path.join(BASE_DIR, "data/italy_dem_combined.tiff")
RADAR_FILEPATH = "/storage/external_01/hiwefi/data/rdr0_val_previews/epoch_000/predictions/rdr0_d01_20250903Z1110_VMI_pred.tiff"
REPROJECTED_RADAR_FILEPATH = os.path.join(BASE_DIR, "data/reprojected_radar.tiff")
CN_MAP_FILEPATH = os.path.join(BASE_DIR, "data/glc_italy.tif")
D8_FILEPATH = os.path.join(BASE_DIR, "D8_output.tiff")
ALIGNED_CN_FILEPATH = os.path.join(BASE_DIR, "aligned_cn.tif")
'''
'''
# Percorsi dei file
DEM_FILEPATH = "data/campania_dem_combined.tiff"
RADAR_FILEPATH = "data/rdr0_d01_20250401Z0540_VMI_cropped.tiff" # dinosauro
# RADAR_FILEPATH = "rdr0_d01_20250314Z0710_VMI.tiff"
RADAR_FILES = sorted(glob.glob("path_to_radar_tiffs/*.tif"))
RADAR_CAMPANIA = "data/cropped.tif"
REPROJECTED_RADAR_FILEPATH = "data/reprojected_radar.tiff"
CN_MAP_FILEPATH = "data/aligned_cn.tif"
D8_FILEPATH = "D8_output.tiff"
ALIGNED_CN_FILEPATH = "aligned_cn.tif"
'''

# Cartella base (dove si trova questo file)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Path al DEM combinato dell’Italia
DEM_FILEPATH = os.path.join(DATA_DIR, "italy_dem_combined.tiff")

# Path alla mappa di copertura del suolo/Curve Number per l’Italia (convertita da C3S)
CN_MAP_FILEPATH = os.path.join(DATA_DIR, "glc_italy.tif")

STEP_MINUTES = 10  # Intervallo temporale tra le predizioni radar in minuti 
WINDOW_STEPS = 6  # Numero di step temporali da considerare per il calcolo del runoff

# Path alla mappa Curve Number riallineata sulla griglia del DEM
ALIGNED_CN_FILEPATH = os.path.join(DATA_DIR, "aligned_cn_italy.tif")

# Directory che contiene i file GeoTIFF delle precipitazioni predette dal modello AI
PREDICTION_DIR = "/home/vbucciero/projects/flood_risk_model/dataset/predictions"
PREDICTION_FILE = os.path.join(PREDICTION_DIR, "rdr0_d01_20250725Z1720_VMI_pred.tiff")  # Nuovo file random per test

PREDICTION_START_FILE = PREDICTION_FILE

# Path per il file di direzione del flusso D8 (generato dallo script hydrology)
D8_FILEPATH = os.path.join(DATA_DIR, "D8_output_italy.tiff")
