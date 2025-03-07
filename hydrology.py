import rasterio
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
from raster_utils import sea_mask

# def compute_runoff(precipitation, cn_map, mask):
#     """
#     Calcola il deflusso superficiale basato sul metodo SCS-CN.
    
#     :param precipitation: Array numpy contenente i dati di precipitazione.
#     :param cn_map: Array numpy contenente la mappa Curve Number.
#     :return: Array numpy contenente il deflusso superficiale (Runoff).
#     """
#     cn_map = np.where(cn_map <= 0 | (mask == 0), 1e-6, cn_map)  # Evita divisione per zero
    
#     # Calcola la capacità di ritenzione S
#     S = (1000 / cn_map) - 10
#     S = np.where(S < 0, 0, S)  
    
#     # Gestione dei valori NaN o negativi nella precipitazione
#     precipitation = np.where(precipitation <= 0, 0, precipitation)
    
#     # Calcola il deflusso superficiale
#     runoff = np.zeros_like(precipitation, dtype=np.float32)
#     valid_mask = precipitation > 0.2 * S  # Maschera per le celle valide
    
#     # Evita divisioni per zero
#     denominator = precipitation + 0.8 * S
#     denominator = np.where(denominator == 0, 1e-6, denominator)  # Evita divisione per zero
    
#     runoff[valid_mask] = ((precipitation[valid_mask] - 0.2 * S[valid_mask]) ** 2) / denominator[valid_mask]
    
#     return runoff

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

    S = np.maximum((1000 / cn_map) - 10, 0)

    runoff = np.zeros_like(precipitation, dtype=np.float32)
    valid_mask = (precipitation > 0.2 * S) & (mask == 1)  

    runoff[valid_mask] = ((precipitation[valid_mask] - 0.2 * S[valid_mask]) ** 2) / (
        precipitation[valid_mask] + 0.8 * S[valid_mask] + 1e-6
    )
    
    return runoff * mask

# Funzione per generare una mappa del rischio di alluvione non utilizzata
def generate_flood_risk_map(dem, runoff, flow_accumulation):
    """
    Combina DEM, runoff e flow accumulation per generare una mappa del rischio.
    """
    flood_risk = (runoff * flow_accumulation) / (dem + 1)
    flood_risk = np.clip(flood_risk, 0, 1)
    return flood_risk

# Funzione di avviso di alluvione non utilizzata
def flood_alert(flood_risk):
    """
    Se il rischio supera una soglia critica, genera un avviso di alluvione.
    """
    if np.max(flood_risk) > 0.8:
        print("🚨 ATTENZIONE: Alto rischio di alluvione! 🚨")
    else:
        print("✅ Nessun rischio di alluvione significativo.")
        
# calcolo D8
D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
D8_VALUES = [128, 1, 2, 4, 8, 16, 32, 64]

def calculate_flow_direction(window_data, mask):
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
    window, data, mask = args
    return (window, calculate_flow_direction(data, mask))

def calculate_flow_direction_parallel(tiff_path, output_path, mask):
    """
    Calcola il d8.
    """
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
                    
# def calculate_flood_risk(d8_filepath, runoff):
#     with rasterio.open(d8_filepath) as d8_src:
#         d8_flow = d8_src.read(1)
#     flood_risk_map = d8_flow * runoff
#     return flood_risk_map

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
  
  
# def compute_flood_risk(runoff, flow_direction):
#     """
#     Combina runoff e direzione del flusso per creare una mappa di rischio alluvionale.
    
#     :param runoff: Array numpy con il deflusso superficiale normalizzato.
#     :param flow_direction: Array numpy con la direzione del flusso normalizzata.
#     :return: Mappa del rischio alluvionale.
#     """
#     runoff_norm = normalize(runoff)
#     flow_norm = normalize(flow_direction)
    
#     # Ponderazione: il runoff pesa di più rispetto alla direzione del flusso
#     risk_map = (0.7 * runoff_norm) + (0.3 * flow_norm)
    
#     # Applica un filtro gaussiano per rendere la mappa più fluida
#     risk_map = gaussian_filter(risk_map, sigma=2)
    
#     return risk_map