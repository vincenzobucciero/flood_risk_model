'''
import glob
import os

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

import os

# Cartella base (dove si trova questo file)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "/home/v.bucciero/data/instruments/rdr0_previews_h100gpu/epoch_000/predictions")

# Path al DEM combinato dell’Italia
DEM_FILEPATH = os.path.join(DATA_DIR, "italy_dem_combined.tiff")

# Path alla mappa di copertura del suolo/Curve Number per l’Italia (convertita da C3S)
CN_MAP_FILEPATH = os.path.join(DATA_DIR, "glc_italy.tif")

# Path alla mappa Curve Number riallineata sulla griglia del DEM
ALIGNED_CN_FILEPATH = os.path.join(DATA_DIR, "aligned_cn_italy.tif")

# Directory che conterrà i file GeoTIFF delle precipitazioni predette dal tuo modello AI.
# Aggiorna questo percorso con la cartella dove salverai le tue predizioni.
PREDICTION_DIR = os.path.join(DATA_DIR, "precip_predictions")

# Path per il file di direzione del flusso D8 (generato dallo script hydrology)
D8_FILEPATH = os.path.join(DATA_DIR, "D8_output_italy.tiff")
