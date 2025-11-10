'''
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
from mpi4py import MPI
import subprocess

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

def add_ghost_cells(dem):
    rows, cols = dem.shape
    local_dem = np.zeros((rows + 2, cols + 2))
    local_dem[1:-1, 1:-1] = dem
    
    return local_dem

def exchange_ghost_cells(local_matrix, rank, size, num_block):
    comm = MPI.COMM_WORLD
    up, down, left, right = rank - num_block, rank + num_block, rank - 1, rank + 1
    
    if up >= 0:
        print(f"Rank {rank}: Sending to up ({up}) and receiving from up")
        comm.Sendrecv(local_matrix[1, :], dest=up, recvbuf=local_matrix[0, :], source=up)
    if down < size:
        print(f"Rank {rank}: Sending to down ({down}) and receiving from down")
        comm.Sendrecv(local_matrix[-2, :], dest=down, recvbuf=local_matrix[-1, :], source=down)
        
    if left % num_block != num_block - 1:
        print(f"Rank {rank}: Sending to left ({left}) and receiving from left")
        comm.Sendrecv(local_matrix[:, 1], dest=left, recvbuf=local_matrix[:, 0], source=left)
    if right % num_block != 0:
        print(f"Rank {rank}: Sending to right ({right}) and receiving from right")
        comm.Sendrecv(local_matrix[:, -2], dest=right, recvbuf=local_matrix[:, -1], source=right)
        
    return local_matrix

def compute_d8(elev):
    d8 = np.zeros_like(elev, dtype=np.uint8)
    
    directions = [16, 8, 4, 2, 1, 32, 64, 128]
    shift = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    
    for i in range(1, elev.shape[0] - 1):
        for j in range(1, elev.shape[1] -1):
            max_slope = 0
            max_dir = 0
            for k, (di, dj) in enumerate(shift):
                ni, nj = i + di, j + dj
                slope = (elev[i, j] - elev[ni, nj]) / (np.sqrt(2) if di != 0 and dj != 0 else 1)
                if slope > max_slope:
                    max_slope = slope
                    max_dir = directions[k]
            d8[i, j] = max_dir
    return d8

def process_d8(rank, size, num_block, dem, profile):
    block_rows = dem.shape[0] // num_block
    block_cols = dem.shape[1] // num_block
    
    row_start = (rank // num_block) * block_rows
    row_end = row_start + block_rows
    col_start = (rank % num_block) * block_cols
    col_end = col_start + block_cols
    
    local_dem = dem[row_start:row_end, col_start:col_end]
    local_dem = add_ghost_cells(local_dem)
    
    local_dem = exchange_ghost_cells(local_dem, rank, size, num_block)
    
    local_d8 = compute_d8(local_dem)
    final_d8 = local_d8[1:-1, 1:-1]
    
    return final_d8

def d8_initialize(dem, d8_tiff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Rank {rank} of {size} started processing.")
    
    start_time = None
    if rank == 0:
        start_time = time.perf_counter()
    
    with rasterio.open(dem) as src:
        dem = src.read(1)
        profile = src.profile
        
    num_block = int(np.sqrt(size))
    if num_block * num_block != size:
        num_block = 1
    
    final_d8 = process_d8(rank, size, num_block, dem, profile)
    
    gathered_d8 = None
    if rank == 0:
        gathered_d8 = np.zeros_like(dem)
        
    final_d8 = np.ascontiguousarray(final_d8)  

    if rank == 0:
        gathered_d8 = np.zeros_like(dem) 
        gathered_d8 = np.ascontiguousarray(gathered_d8)  

    comm.Gather(final_d8, gathered_d8, root=0)
    
    if rank == 0:
        with rasterio.open(d8_tiff, "w", **profile) as dst:
            dst.write(gathered_d8, 1)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
  
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
'''

import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
from raster_utils import sea_mask, align_radar_to_dem
from tempfile import NamedTemporaryFile
from termcolor import colored
from netCDF4 import Dataset
import xarray as xr
import time
from datetime import datetime
from mpi4py import MPI
import subprocess

def compute_runoff(precipitation, cn_map, mask):
    """"
    Calcola il deflusso superficiale basato sul metodo SCS-CN.
    
    :param precipitation: Array numpy contenente i dati di precipitazione.
    :param cn_map: Array numpy contenente la mappa Curve Number.
    :param mask: Maschera che identifica dove c'è terreno e dove c'è il mare.
    :return: Array numpy contenente il deflusso superficiale (Runoff).
    """
    cn_map = np.where((cn_map <= 0) | (mask == 0), 1e-6, cn_map)  
    precipitation = np.where(mask == 0, 0, precipitation) 

    S = np.maximum((1000 / cn_map) - 10, 0)  # capacità di ritenzione potenziale

    runoff = np.zeros_like(precipitation, dtype=np.float32)
    valid_mask = (precipitation > 0.2 * S) & (mask == 1)  

    runoff[valid_mask] = ((precipitation[valid_mask] - 0.2 * S[valid_mask]) ** 2) / (
        precipitation[valid_mask] + 0.8 * S[valid_mask] + 1e-6)
    
    return runoff * mask

def add_ghost_cells(dem):
    rows, cols = dem.shape
    local_dem = np.zeros((rows + 2, cols + 2))
    local_dem[1:-1, 1:-1] = dem
    return local_dem

def exchange_ghost_cells(local_matrix, rank, size, num_block):
    comm = MPI.COMM_WORLD
    up, down, left, right = rank - num_block, rank + num_block, rank - 1, rank + 1
    
    if up >= 0:
        print(f"Rank {rank}: Sending to up ({up}) and receiving from up")
        comm.Sendrecv(local_matrix[1, :], dest=up, recvbuf=local_matrix[0, :], source=up)
    if down < size:
        print(f"Rank {rank}: Sending to down ({down}) and receiving from down")
        comm.Sendrecv(local_matrix[-2, :], dest=down, recvbuf=local_matrix[-1, :], source=down)
        
    if left % num_block != num_block - 1:
        print(f"Rank {rank}: Sending to left ({left}) and receiving from left")
        comm.Sendrecv(local_matrix[:, 1], dest=left, recvbuf=local_matrix[:, 0], source=left)
    if right % num_block != 0:
        print(f"Rank {rank}: Sending to right ({right}) and receiving from right")
        comm.Sendrecv(local_matrix[:, -2], dest=right, recvbuf=local_matrix[:, -1], source=right)
        
    return local_matrix

def compute_d8(elev):
    d8 = np.zeros_like(elev, dtype=np.uint8)
    
    directions = [16, 8, 4, 2, 1, 32, 64, 128]
    shift = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    
    for i in range(1, elev.shape[0] - 1):
        for j in range(1, elev.shape[1] - 1):
            max_slope = 0
            max_dir = 0
            for k, (di, dj) in enumerate(shift):
                ni, nj = i + di, j + dj
                slope = (elev[i, j] - elev[ni, nj]) / (np.sqrt(2) if di != 0 and dj != 0 else 1)
                if slope > max_slope:
                    max_slope = slope
                    max_dir = directions[k]
            d8[i, j] = max_dir
    return d8

def process_d8(rank, size, num_block, dem, profile):
    block_rows = dem.shape[0] // num_block
    block_cols = dem.shape[1] // num_block
    
    row_start = (rank // num_block) * block_rows
    row_end = row_start + block_rows
    col_start = (rank % num_block) * block_cols
    col_end = col_start + block_cols
    
    local_dem = dem[row_start:row_end, col_start:col_end]
    local_dem = add_ghost_cells(local_dem)
    
    local_dem = exchange_ghost_cells(local_dem, rank, size, num_block)
    
    local_d8 = compute_d8(local_dem)
    final_d8 = local_d8[1:-1, 1:-1]
    
    return final_d8

def d8_initialize(dem, d8_tiff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Rank {rank} of {size} started processing.")
    
    start_time = None
    if rank == 0:
        start_time = time.perf_counter()
    
    with rasterio.open(dem) as src:
        dem_data = src.read(1)
        profile = src.profile
        
    num_block = int(np.sqrt(size))
    if num_block * num_block != size:
        num_block = 1
    
    final_d8 = process_d8(rank, size, num_block, dem_data, profile)
    
    gathered_d8 = None
    if rank == 0:
        gathered_d8 = np.zeros_like(dem_data)
        
    final_d8 = np.ascontiguousarray(final_d8)  

    if rank == 0:
        gathered_d8 = np.zeros_like(dem_data) 
        gathered_d8 = np.ascontiguousarray(gathered_d8)  

    comm.Gather(final_d8, gathered_d8, root=0)
    
    if rank == 0:
        with rasterio.open(d8_tiff, "w", **profile) as dst:
            dst.write(gathered_d8, 1)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

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
    """Visualizza la mappa del rischio."""
    plt.figure(figsize=(10, 6))
    plt.imshow(np.where(mask == 1, flood_risk_map, np.nan), cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Flood Risk")
    plt.title("Flood Risk Map")
    plt.show()

def get_radar_files(directory):
    """
    Restituisce una lista di percorsi completi dei file TIFF nella directory specificata.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tiff")]

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
            tolerance = 1e-1  # Tolleranza per gestire piccole variazioni
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
    Processa un singolo file di precipitazione:
    lo riallinea al DEM e calcola il runoff.
    """
    # Step 1: Allinea il file direttamente al DEM
    with NamedTemporaryFile(suffix=".tiff", delete=False) as temp_aligned:
        aligned_tiff = temp_aligned.name
    try:
        align_radar_to_dem(file, dem_tiff, aligned_tiff)
    except Exception as e:
        print(f"Errore durante l'allineamento del file {file}: {e}")
        return None

    # Step 2: Calcola il runoff dal file allineato
    try:
        runoff = process_tiff(aligned_tiff, cn_map, mask)
    except Exception as e:
        print(f"Errore durante il calcolo del runoff per il file {aligned_tiff}: {e}")
        return None

    # Pulizia del file temporaneo
    os.remove(aligned_tiff)
    return runoff

def calculate_accumulated_runoff(radar_directory, cn_map, mask, dem_tiff, output_path, output_format="tiff"):
    """
    Calcola il runoff accumulato su più file radar/predizioni, dopo averli allineati al DEM.
    Permette di salvare il risultato in formato TIFF o NetCDF a seconda del parametro 'output_format'.
    
    Parametri:
    - radar_directory: Cartella contenente i file di precipitazione
    - cn_map: Mappa Curve Number
    - mask: Maschera (mare/terra)
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
    
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)

'''
def save_as_netcdf(data, output_path, reference_file, variable_name="value", timestamp=None):
    """
    Salva un array NumPy come file NetCDF (.nc), includendo latitudine, longitudine e timestamp.
    Calcola e restituisce anche le statistiche dei dati (min, media, max).
    
    Args:
        data: Array 2D da salvare
        output_path: Percorso del file di output
        reference_file: File di riferimento per la georeferenziazione
        variable_name: Nome della variabile principale (default: "value")
        timestamp: Timestamp in formato stringa (es. "20250611Z1630") o oggetto datetime
    
    Returns:
        tuple: (min, media, max) dei dati
    """
    nrows, ncols = data.shape
    
    # Ottieni la georeferenziazione dal file di riferimento
    with rasterio.open(reference_file) as src:
        transform = src.transform
        x_min, y_max = transform * (0, 0)  # Coordinate del pixel in alto a sinistra
        x_max, y_min = transform * (ncols, nrows)  # Coordinate del pixel in basso a destra
    
    # Creiamo i vettori latitudine e longitudine
    lon = np.linspace(x_min, x_max, ncols)
    lat = np.linspace(y_max, y_min, nrows)

    with Dataset(output_path, "w", format="NETCDF4") as nc:
        # Crea le dimensioni
        nc.createDimension("time", 1)  # Dimensione singola per il timestamp
        nc.createDimension("longitude", ncols)
        nc.createDimension("latitude", nrows)

        # Crea le variabili di latitudine e longitudine
        time_var = nc.createVariable("time", "i4", ("time",))  # Cambiato in i4 per intero
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        
        
        time_var.description = "Time"
        time_var.long_name = "time"
        time_var.units = "hours since 1900-01-01 00:00:0.0"

        lon_var.description = "Longitude"
        lon_var.units = "degrees_east"

        lat_var.description = "Latitude"
        lat_var.units = "degrees_north"
        
        lat_var[:] = lat
        lon_var[:] = lon
        
        # Gestione del timestamp
        if timestamp is not None:
            if isinstance(timestamp, int):
                time_var[0] = timestamp  # Assegna direttamente l'intero
            else:
                # Se è un oggetto datetime, converte in ore dal 1900-01-01
                from datetime import datetime
                ref_date = datetime(1900, 1, 1)
                if isinstance(timestamp, str):
                    # Converte la stringa in datetime
                    timestamp = datetime.strptime(timestamp, "%Y%m%dZ%H%M")
                delta = timestamp - ref_date
                hours = int(delta.total_seconds() / 3600)
                time_var[0] = hours

        # Crea la variabile principale (flood o runoff)
        if variable_name == "runoff":
            main_var = nc.createVariable("runoff", "f4", ("time", "latitude", "longitude"))
            main_var.units = "mm"
            main_var.description = "Surface runoff in millimeters"
        else:  # flood o altri casi
            main_var = nc.createVariable("flood", "f4", ("time", "latitude", "longitude"))
            main_var.units = "normalized_flood"
            main_var.description = "Normalized flood values (0-1)"
        
        # Scrivi i dati
        main_var[0, :, :] = data
        
        # Calcola le statistiche
        data_min = float(np.min(data))
        data_mean = float(np.mean(data))
        data_max = float(np.max(data))
        
        # Crea le variabili per le statistiche con dimensione temporale
        stats_var_name = "runoff_stats" if variable_name == "runoff" else "floodrisk"
        stats_min = nc.createVariable(f"{stats_var_name}_min", "f4", ("time",))
        stats_mean = nc.createVariable(f"{stats_var_name}_mean", "f4", ("time",))
        stats_max = nc.createVariable(f"{stats_var_name}_max", "f4", ("time",))
        
        # Imposta descrizioni e unità
        description = "Runoff statistics" if variable_name == "runoff" else "Flood risk statistics"
        units = "mm" if variable_name == "runoff" else "normalized_flood"
        
        stats_min.description = f"Minimum value of {description}"
        stats_min.units = units
        stats_mean.description = f"Mean value of {description}"
        stats_mean.units = units
        stats_max.description = f"Maximum value of {description}"
        stats_max.units = units
        
        # Assegna i valori
        stats_min[0] = data_min
        stats_mean[0] = data_mean
        stats_max[0] = data_max
        
        # Crea un dizionario con i valori delle statistiche per retrocompatibilità
        stats = {
            'min': float(stats_min[0]),
            'mean': float(stats_mean[0]),
            'max': float(stats_max[0])
        }
        
        return {
            variable_name: main_var[:],
            f"{stats_var_name}": stats
        }
'''

def save_as_netcdf(data, output_path, reference_file, variable_name="runoff", timestamp=None):
    """
    Salva un array NumPy come file NetCDF (.nc) in formato CF-like, assicurando che la
    variabile principale sia geo2D/geo3D (time, latitude, longitude) e che siano scritte
    le statistiche pixel-wise (min/mean/max) come variabili 2D (latitude, longitude).

    Parametri:
        data: array 2D (y,x) oppure 3D (time,y,x). Se 2D viene convertito a (1,y,x).
        output_path: percorso del file di output (es. "out.nc")
        reference_file: file raster (tiff) di riferimento per georeferenziazione
        variable_name: nome della variabile principale (es. "runoff" o "floodrisk")
        timestamp: None, singolo valore (str/int/datetime) o lista di valori con lunghezza == time
                   - se None verrà impostata una dimensione time con valori placeholder 0..nt-1
                   - per stringhe formato accettato: "YYYYmmddZHHMM" oppure ISO parsable

    Ritorna:
        dict con chiavi:
          - "pixel_min", "pixel_mean", "pixel_max": numpy array 2D delle statistiche per pixel
          - "output_path": percorso del file scritto
    """

    # Normalizza la forma dei dati: 2D -> 3D con dimensione time = 1
    arr = np.asarray(data)
    if arr.ndim == 2:
        data3 = arr[np.newaxis, :, :].astype(np.float32)
    elif arr.ndim == 3:
        data3 = arr.astype(np.float32)
    else:
        raise ValueError("`data` deve essere 2D (y,x) o 3D (time,y,x)")

    nt, nrows, ncols = data3.shape

    # Ottieni georeferenziazione e CRS dal file di riferimento
    with rasterio.open(reference_file) as src:
        transform = src.transform
        crs = src.crs
        x_min, y_max = transform * (0, 0)
        x_max, y_min = transform * (ncols, nrows)

    # Creiamo i vettori lon/lat dal transform (lineari)
    lon = np.linspace(x_min, x_max, ncols, dtype=np.float32)
    lat = np.linspace(y_max, y_min, nrows, dtype=np.float32)

    # Helper per convertire timestamp in "hours since 1900-01-01"
    def to_hours_since_1900(t):
        ref_date = datetime(1900, 1, 1)
        if isinstance(t, int):
            return int(t)
        if isinstance(t, str):
            # prova formato YYYYmmddZHHMM altrimenti ISO
            try:
                dt = datetime.strptime(t, "%Y%m%dZ%H%M")
            except Exception:
                dt = datetime.fromisoformat(t)
        elif isinstance(t, datetime):
            dt = t
        else:
            dt = datetime.fromisoformat(str(t))
        return int((dt - ref_date).total_seconds() / 3600)

    # Apri NetCDF e scrivi i dati in formato CF-like
    with Dataset(output_path, "w", format="NETCDF4") as nc:
        # Dimensioni
        nc.createDimension("time", nt)
        nc.createDimension("latitude", nrows)
        nc.createDimension("longitude", ncols)

        # Variabili coordinate
        time_var = nc.createVariable("time", "i4", ("time",))
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))

        # Attributi delle coordinate
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"

        lat_var[:] = lat
        lon_var[:] = lon

        # Gestione della variabile time
        time_var.units = "hours since 1900-01-01 00:00:0.0"
        time_var.long_name = "time"
        if timestamp is None:
            time_var[:] = np.arange(nt, dtype=np.int32)
        else:
            # se timestamp è una lista/array con lunghezza nt -> converti
            if hasattr(timestamp, "__len__") and not isinstance(timestamp, (str, bytes)):
                if len(timestamp) != nt:
                    raise ValueError("Lunghezza di `timestamp` diversa dalla dimensione time dei dati")
                time_var[:] = np.array([to_hours_since_1900(t) for t in timestamp], dtype=np.int32)
            else:
                base = to_hours_since_1900(timestamp)
                time_var[:] = np.arange(base, base + nt, dtype=np.int32)

        # Variabile principale (time, latitude, longitude)
        main_var = nc.createVariable(variable_name, "f4", ("time", "latitude", "longitude"), zlib=True, complevel=4)
        # Impostazione units: compatibile con il tuo uso (runoff in mm, floodrisk normalizzato)
        if variable_name.lower() in ("floodrisk", "flood"):
            main_var.units = "normalized_flood"
        else:
            main_var.units = "mm"
        main_var.long_name = f"{variable_name} per pixel"
        main_var.coordinates = "latitude longitude"

        # Aggiungi info CRS se disponibile (scrive WKT come variabile 'crs' se possibile)
        if crs is not None:
            try:
                crs_var = nc.createVariable("crs", "i4")
                crs_var.spatial_ref = crs.to_wkt()
                main_var.grid_mapping = "crs"
            except Exception:
                # non critico se fallisce
                pass

        # Scrivi i dati (ordine: time, latitude, longitude)
        main_var[:, :, :] = data3

        # Calcola statistiche pixel-wise lungo la dimensione time
        pix_min = np.min(data3, axis=0).astype(np.float32)
        pix_mean = np.mean(data3, axis=0).astype(np.float32)
        pix_max = np.max(data3, axis=0).astype(np.float32)

        # Scrivi le statistiche come variabili 2D (latitude, longitude)
        vmin = nc.createVariable(f"{variable_name}_min", "f4", ("latitude", "longitude"), zlib=True, complevel=4)
        vmean = nc.createVariable(f"{variable_name}_mean", "f4", ("latitude", "longitude"), zlib=True, complevel=4)
        vmax = nc.createVariable(f"{variable_name}_max", "f4", ("latitude", "longitude"), zlib=True, complevel=4)

        units = main_var.units
        vmin.units = units
        vmean.units = units
        vmax.units = units

        vmin.long_name = f"Pixel-wise minimum of {variable_name} over time"
        vmean.long_name = f"Pixel-wise mean of {variable_name} over time"
        vmax.long_name = f"Pixel-wise maximum of {variable_name} over time"

        vmin[:, :] = pix_min
        vmean[:, :] = pix_mean
        vmax[:, :] = pix_max

    # Ritorna le statistiche (pixel-wise) insieme al percorso del file creato
    return {
        "pixel_min": pix_min,
        "pixel_mean": pix_mean,
        "pixel_max": pix_max,
        "output_path": output_path
    }