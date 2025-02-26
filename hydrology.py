import rasterio
import numpy as np
import matplotlib.pyplot as plt

def compute_runoff(precipitation, cn_map):
    """
    Calcola il deflusso superficiale basato sul metodo SCS-CN.
    
    :param precipitation: Array numpy contenente i dati di precipitazione.
    :param cn_map: Array numpy contenente la mappa Curve Number.
    :return: Array numpy contenente il deflusso superficiale (Runoff).
    """
    # Gestione dei valori NaN o negativi nel CN map
    cn_map = np.where(cn_map <= 0, 1e-6, cn_map)  # Evita divisione per zero
    
    # Calcola la capacità di ritenzione S
    S = (1000 / cn_map) - 10
    S = np.where(S < 0, 0, S)  # Imposta S a 0 se negativo
    
    # Gestione dei valori NaN o negativi nella precipitazione
    precipitation = np.where(precipitation <= 0, 0, precipitation)
    
    # Calcola il deflusso superficiale
    runoff = np.zeros_like(precipitation, dtype=np.float32)
    valid_mask = precipitation > 0.2 * S  # Maschera per le celle valide
    
    # Evita divisioni per zero
    denominator = precipitation + 0.8 * S
    denominator = np.where(denominator == 0, 1e-6, denominator)  # Evita divisione per zero
    
    runoff[valid_mask] = ((precipitation[valid_mask] - 0.2 * S[valid_mask]) ** 2) / denominator[valid_mask]
    
    return runoff

# Funzione per generare una mappa del rischio di alluvione
def generate_flood_risk_map(dem, runoff, flow_accumulation):
    """
    Combina DEM, runoff e flow accumulation per generare una mappa del rischio.
    """
    flood_risk = (runoff * flow_accumulation) / (dem + 1)
    flood_risk = np.clip(flood_risk, 0, 1)
    return flood_risk

# Funzione di avviso di alluvione
def flood_alert(flood_risk):
    """
    Se il rischio supera una soglia critica, genera un avviso di alluvione.
    """
    if np.max(flood_risk) > 0.8:
        print("🚨 ATTENZIONE: Alto rischio di alluvione! 🚨")
    else:
        print("✅ Nessun rischio di alluvione significativo.")