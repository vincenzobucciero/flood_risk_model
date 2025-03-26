import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
from raster_utils import sea_mask
from tempfile import NamedTemporaryFile
from raster_utils import crop_tiff_to_campania, align_radar_to_dem
from termcolor import colored
from netCDF4 import Dataset
import xarray as xr
import time

def compute_runoff(precipitation, cn_map, mask):
    """"
    Calcola il deflusso superficiale basato sul metodo SCS-CN.
    
    :param precipitation: Array numpy contenente i dati di precipitazione.
    :param cn_map: Array numpy contenente la mappa Curve Number.
    :param mask: Maschera che identifica dove c'è terreno e dove c'è il mare
    :return: Array numpy contenente il deflusso superficiale (Runoff).
    """
    cn_map = np.where((cn_map <= 0) | (mask == 0), 1e-6, cn_map)  
    precipitation = np.where(mask == 0, 0, precipitation) 

    S = np.maximum((1000 / cn_map) - 10, 0) # capacità di ritenzione potenziale

    runoff = np.zeros_like(precipitation, dtype=np.float32)
    valid_mask = (precipitation > 0.2 * S) & (mask == 1)  

    runoff[valid_mask] = ((precipitation[valid_mask] - 0.2 * S[valid_mask]) ** 2) / (
        precipitation[valid_mask] + 0.8 * S[valid_mask] + 1e-6)
    
    return runoff * mask

# calcolo D8
D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
D8_VALUES = [128, 1, 2, 4, 8, 16, 32, 64]

def calculate_flow_direction(window_data, mask):
    """
    Calcola la direzione del flusso usando il metodo D8.
    """
    rows, cols = window_data.shape
    flow_dir = np.zeros_like(window_data, dtype=np.uint8)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if mask[r, c] == 0:  # Se è mare, salta il calcolo
                continue  

            center = window_data[r, c]
            if np.isnan(center):
                continue

            min_diff = float("inf")
            min_dir = 0

            for i, (dr, dc) in enumerate(D8_OFFSETS):
                neighbor = window_data[r + dr, c + dc]
                
                # Considera solo celle sulla terraferma
                if not np.isnan(neighbor) and mask[r + dr, c + dc] == 1:
                    diff = center - neighbor
                    if diff > 0 and diff < min_diff:
                        min_diff = diff
                        min_dir = D8_VALUES[i]

            flow_dir[r, c] = min_dir  

    return flow_dir[1:-1, 1:-1]  

def process_window(args):
    """
    Wrapper per parallelizzare il calcolo della direzione del flusso.
    """
    window, data, mask = args
    return (window, calculate_flow_direction(data, mask))

def calculate_flow_direction_parallel(tiff_path, output_path, mask):
    """
    Calcola la direzione del flusso D8 in parallelo.
    """
    start_time = time.time()
    
    with rasterio.open(tiff_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8)

        windows = [window for _, window in src.block_windows()]
        data = [src.read(1, window=window) for window in windows]

        with Pool(cpu_count()) as pool:
            results = pool.map(process_window, zip(windows, data, [mask]*len(windows)))

        with rasterio.open(output_path, 'w', **profile) as dst:
            for window, result in results:
                dst.write(result, 1, window=window)    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    with open("scalability_test.txt", "a") as log:
        log.write(f"Calcolo direzione del flusso: {elapsed_time:.2f} secondi\n")
    print(f"Calcolo direzione del flusso completato in {elapsed_time:.2f} secondi")
    
def scalability_test(tiff_path, output_path, mask, max_cpus=16, log_file="scalability_test.txt"):
    """
    Testa la scalabilità del calcolo della direzione del flusso variando il numero di CPU.
    """
    with open(log_file, "w") as log:
        log.write("Scalability Test: Testing with varying numbers of CPUs.\n")
    
    for cpus in range(1, max_cpus + 1):
        start_time = time.time()
        with Pool(cpus) as pool:
            # Ripeti l'operazione per testare la scalabilità con differenti CPU
            with rasterio.open(tiff_path) as src:
                profile = src.profile.copy()
                profile.update(dtype=rasterio.uint8)

                windows = [window for _, window in src.block_windows()]
                data = [src.read(1, window=window) for window in windows]

                results = pool.map(process_window, zip(windows, data, [mask] * len(windows)))

                with rasterio.open(output_path, 'w', **profile) as dst:
                    for window, result in results:
                        dst.write(result, 1, window=window)

        end_time = time.time()
        elapsed_time = end_time - start_time

        with open(log_file, "a") as log:
            log.write(f"Using {cpus} CPUs: {elapsed_time:.2f} seconds\n")
        
        print(f"Using {cpus} CPUs: {elapsed_time:.2f} seconds")
                    
def normalize(array):
    """Normalizza un array tra 0 e 1."""
    return (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-6)

def compute_flood_risk(flow_direction_tiff, runoff):
    """Combina direzione del flusso e runoff per calcolare il rischio di alluvione."""
    with rasterio.open(flow_direction_tiff) as src:
        flow_direction = src.read(1).astype(np.float32)  

    runoff_norm = normalize(runoff.astype(np.float32)) 
    flow_norm = normalize(flow_direction)

    risk_map = (0.7 * runoff_norm) + (0.3 * flow_norm)
    return gaussian_filter(risk_map, sigma=2)

def visualize_flood_risk(flood_risk_map, mask):
    """Visualizza la mappa del rischio"""
    plt.figure(figsize=(10, 6))
    plt.imshow(np.where(mask == 1, flood_risk_map, np.nan), cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Flood Risk")
    plt.title("Flood Risk Map")
    plt.show()
  
def get_radar_files(directory):
    """
    Restituisce una lista di percorsi completi dei file TIFF nella directory specificata.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tiff')]

def process_tiff(file, cn_map, mask):
    """
    Elabora un file TIFF per calcolare il runoff.
    """
    with rasterio.open(file) as src:
        precipitation = src.read(1)
        
        # Leggi il valore nodata dai metadati
        nodata = src.nodata
        print(f"Valore nodata dichiarato nel file {file}: {nodata}")
        
        # Sostituisci i valori nodata con 0, usando una tolleranza
        if nodata is not None:
            tolerance = 1e-1  # Aumentiamo la tolleranza per gestire piccole variazioni
            precipitation = np.where(
                np.isclose(precipitation, nodata, atol=tolerance),
                0,
                precipitation
            )
        
        # Verifica la presenza di valori negativi o anomali
        if np.any(precipitation < 0):
            print("⚠️ Attenzione: Rilevati valori negativi nella precipitazione.")
            precipitation[precipitation < 0] = 0  # Forza i valori negativi a 0
        
        # Verifica la presenza di valori validi
        if np.max(precipitation) <= 0:
            print("⚠️ Attenzione: Nessun valore valido di precipitazione trovato.")
        
        print(f"Valori di precipitazione (dopo la correzione): Min={np.min(precipitation)}, Max={np.max(precipitation)}, Media={np.mean(precipitation)}")
        
        # Calcola il runoff
        runoff = compute_runoff(precipitation, cn_map, mask)
    
    # Sostituisci NaN con 0
    runoff = np.nan_to_num(runoff, nan=0.0)
    
    # Verifica ulteriore per valori negativi nel runoff
    if np.any(runoff < 0):
        print("⚠️ Attenzione: Rilevati valori negativi nel runoff.")
        runoff[runoff < 0] = 0  # Forza i valori negativi a 0
    
    return runoff

def process_radar_file(file, cn_map, mask, dem_tiff):
    """
    Processa un singolo file radar: crop, align, e calcolo del runoff.
    """
    # Step 1: Crop del file radar sull'area della Campania
    with NamedTemporaryFile(suffix=".tiff", delete=False) as temp_crop:
        cropped_tiff = temp_crop.name
    try:
        crop_tiff_to_campania(file, cropped_tiff)
    except Exception as e:
        print(f"Errore durante il cropping del file {file}: {e}")
        return None

    # Step 2: Allinea il file croppato al DEM
    with NamedTemporaryFile(suffix=".tiff", delete=False) as temp_aligned:
        aligned_tiff = temp_aligned.name
    try:
        align_radar_to_dem(cropped_tiff, dem_tiff, aligned_tiff)
    except Exception as e:
        print(f"Errore durante l'allineamento del file {cropped_tiff}: {e}")
        return None

    # Step 3: Calcola il runoff dal file allineato
    try:
        runoff = process_tiff(aligned_tiff, cn_map, mask)
    except Exception as e:
        print(f"Errore durante il calcolo del runoff per il file {aligned_tiff}: {e}")
        return None

    # Pulizia dei file temporanei
    os.remove(cropped_tiff)
    os.remove(aligned_tiff)

    return runoff

# def calculate_accumulated_runoff(radar_directory, cn_map, mask, dem_tiff, output_path):
#     """
#     Calcola il runoff accumulato su più file radar, dopo averli croppati e allineati.
#     """
#     accumulated_runoff = None
    
#     # Ottieni la lista dei file radar
#     radar_files = get_radar_files(radar_directory)
    
#     # Lista per memorizzare i runoff intermedi
#     individual_runoffs = []

#     for file in radar_files:
#         if not os.path.exists(file):
#             print(f"File non trovato: {file}")
#             continue
        
#         # Processa il file radar
#         runoff = process_radar_file(file, cn_map, mask, dem_tiff)
#         if runoff is None:
#             print(f"Errore nel processamento del file: {file}")
#             continue
        
#         # Salva il runoff individuale
#         individual_runoffs.append(runoff)
        
#         # Aggiorna il runoff accumulato
#         if accumulated_runoff is None:
#             accumulated_runoff = np.zeros_like(runoff)
#         elif runoff.shape != accumulated_runoff.shape:
#             print(f"Errore: Dimensioni incompatibili tra runoff ({runoff.shape}) e accumulated_runoff ({accumulated_runoff.shape})")
#             return None
        
#         accumulated_runoff += runoff
#         print(colored(f"Runoff calcolato con successo per il file: {file}", "green"))
    
#     # Mostra i runoff intermedi
#     print("\n--- Runoff intermedi ---")
#     for i, runoff in enumerate(individual_runoffs):
#         print(f"Runoff del file {i+1}: Min={np.min(runoff)}, Max={np.max(runoff)}, Media={np.mean(runoff)}")
    
#     # Mostra il runoff accumulato
#     if accumulated_runoff is not None:
#         print("\n--- Runoff accumulato ---")
#         print(f"Min={np.min(accumulated_runoff)}, Max={np.max(accumulated_runoff)}, Media={np.mean(accumulated_runoff)}")
        
#         # Salva il risultato finale
#         with rasterio.open(radar_files[0]) as src:
#             profile = src.profile.copy()
#             profile.update(dtype=rasterio.float32, count=1)
        
#         with rasterio.open(output_path, 'w', **profile) as dst:
#             dst.write(accumulated_runoff, 1)
        
#         print(f"\nFile runoff accumulato salvato in: {output_path}")
    
#     return accumulated_runoff

def calculate_accumulated_runoff(radar_directory, cn_map, mask, dem_tiff, output_path, output_format="tiff"):
    """
    Calcola il runoff accumulato su più file radar, dopo averli croppati e allineati.
    Permette di salvare il risultato in formato TIFF o NetCDF a seconda del parametro 'output_format'.
    
    Parametri:
    - radar_directory: Cartella contenente i file radar
    - cn_map: Mappa Curve Number
    - mask: Maschera
    - dem_tiff: Modello digitale di elevazione (DEM)
    - output_path: Percorso di output (senza estensione)
    - output_format: Formato di output ("tiff" o "netcdf")
    
    Ritorna:
    - L'array numpy del runoff accumulato
    """
    accumulated_runoff = None
    radar_files = get_radar_files(radar_directory)
    individual_runoffs = []

    for file in radar_files:
        if not os.path.exists(file):
            print(f"File non trovato: {file}")
            continue
        
        runoff = process_radar_file(file, cn_map, mask, dem_tiff)
        if runoff is None:
            print(f"Errore nel processamento del file: {file}")
            continue
        
        individual_runoffs.append(runoff)
        
        if accumulated_runoff is None:
            accumulated_runoff = np.zeros_like(runoff, dtype=np.float32)
        elif runoff.shape != accumulated_runoff.shape:
            print(f"Errore: Dimensioni incompatibili tra runoff ({runoff.shape}) e accumulated_runoff ({accumulated_runoff.shape})")
            return None
        
        accumulated_runoff += runoff
        print(colored(f"Runoff calcolato con successo per il file: {file}", "green"))
    
    # Mostra statistiche
    print("\n--- Runoff intermedi ---")
    for i, runoff in enumerate(individual_runoffs):
        print(f"Runoff del file {i+1}: Min={np.min(runoff)}, Max={np.max(runoff)}, Media={np.mean(runoff)}")

    if accumulated_runoff is not None:
        print("\n--- Runoff accumulato ---")
        print(f"Min={np.min(accumulated_runoff)}, Max={np.max(accumulated_runoff)}, Media={np.mean(accumulated_runoff)}")

        if output_format.lower() == "tiff":
            save_as_tiff(accumulated_runoff, output_path + ".tif", radar_files[0])
        elif output_format.lower() == "netcdf":
            save_as_netcdf(accumulated_runoff, output_path + ".nc", radar_files[0])
        else:
            print("Formato non supportato. Usa 'tiff' o 'netcdf'.")
            return None
        
        print(f"\nFile runoff accumulato salvato in: {output_path}.{output_format.lower()}")
    
    return accumulated_runoff

def save_as_tiff(data, output_path, reference_file):
    """
    Salva un array NumPy come file TIFF, copiando il profilo da un file di riferimento.
    """
    with rasterio.open(reference_file) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

def save_as_netcdf(data, output_path, reference_file):
    """
    Salva un array NumPy come file NetCDF (.nc), includendo latitudine e longitudine.
    """
    nrows, ncols = data.shape
    
    # Ottieni la georeferenziazione dal file di riferimento
    with rasterio.open(reference_file) as src:
        transform = src.transform
        x_min, y_max = transform * (0, 0)  # Coordinate del pixel in alto a sinistra
        x_max, y_min = transform * (ncols, nrows)  # Coordinate del pixel in basso a destra
        res_x, res_y = transform.a, -transform.e  # Risoluzione dei pixel

    # Creiamo i vettori latitudine e longitudine
    lon = np.linspace(x_min, x_max, ncols)
    lat = np.linspace(y_max, y_min, nrows)

    with Dataset(output_path, "w", format="NETCDF4") as nc:
        # Crea le dimensioni
        nc.createDimension("y", nrows)
        nc.createDimension("x", ncols)

        # Crea le variabili di latitudine e longitudine
        lat_var = nc.createVariable("latitude", "f4", ("y",))
        lon_var = nc.createVariable("longitude", "f4", ("x",))

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        lat_var[:] = lat
        lon_var[:] = lon

        # Crea la variabile runoff
        runoff_var = nc.createVariable("runoff", "f4", ("y", "x"))
        runoff_var.units = "mm"  # Unità di misura personalizzabile

        # Scrivi i dati
        runoff_var[:, :] = data