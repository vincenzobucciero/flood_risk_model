#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from termcolor import colored

from raster_utils import *
from hydrology2 import *
import config

# === Dove salvare gli output ===
OUTPUT_DIR = "/storage/external_01/hiwefi/flood_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility per timestamp nei nomi file ---

FNAME_RE = re.compile(r".*?(\d{8})Z(\d{4})_VMI.*?_pred\.tif{1,2}f?$")
# Esempio: rdr0_d01_20251008Z0300_VMI_pred.tiff  -> gruppi: 20251008 e 0300

def parse_ts_from_fname(path):
    """Estrae un datetime dal nome file ..._YYYYMMDDZHHMM_..._pred.tiff."""
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    day, hm = m.groups()
    try:
        return datetime.strptime(day + hm, "%Y%m%d%H%M")
    except Exception:
        return None

def ts_tag(dt: datetime | None) -> str:
    """Ritorna tag 'YYYYMMDDZHHMM' da un datetime (o 'unknown')."""
    return dt.strftime("%Y%m%dZ%H%M") if dt else "unknown"

def get_previous_flood_file(current_timestamp):
    """
    Cerca il file di flood risk precedente (10 minuti prima) nella directory OUTPUT_DIR.
    
    Args:
        current_timestamp (datetime): Il timestamp del file corrente
        
    Returns:
        str or None: Il path del file precedente se esiste, altrimenti None
    """
    prev_ts = current_timestamp - timedelta(minutes=10)
    prev_tag = prev_ts.strftime("%Y%m%dZ%H%M")
    prev_file = os.path.join(OUTPUT_DIR, f"floodrisk_{prev_tag}.nc")
    
    if os.path.exists(prev_file):
        print(colored(f"Found previous flood risk file: {prev_file}", "green"))
        return prev_file
    else:
        print(colored(f"No previous flood risk file found for timestamp {prev_tag}", "yellow"))
        return None

def load_previous_flood(file_path):
    """
    Carica il flood risk dal file NetCDF precedente.
    
    Args:
        file_path (str): Path al file NetCDF
        
    Returns:
        numpy.ndarray: Array 2D con i valori di flood risk
    """
    if file_path is None:
        return None
        
    try:
        ds = xr.open_dataset(file_path, decode_times=False)
        # Prima prova a cercare la variabile con il nome esatto che usiamo per salvare
        if "value" in ds:
            flood = ds["value"].values
        # Altrimenti prova nomi alternativi che potrebbero essere stati usati in precedenza
        elif "flood" in ds:
            flood = ds["flood"].values
        elif "floodrisk" in ds:
            flood = ds["floodrisk"].values
        elif "runoff" in ds:
            flood = ds["runoff"].values
        else:
            raise KeyError(f"Nessuna variabile di flood risk trovata nel file. Variabili disponibili: {list(ds.variables.keys())}")
        ds.close()
        return flood
    except Exception as e:
        print(colored(f"Error loading previous flood risk: {e}", "red"))
        return None

def save_single_step(path, arr, dem_like_path, var_name="value", timestamp=None):
    """
    Salva un array 2D come NetCDF con georeferenziazione ricavata dal dem_like_path.
    Ritorna le statistiche (min, media, max) dei dati.
    
    Args:
        path: percorso del file di output
        arr: array da salvare
        dem_like_path: file di riferimento per la georeferenziazione
        var_name: nome della variabile
        timestamp: datetime opzionale per il timestamp
    """
    return save_as_netcdf(arr.astype(np.float32), path, dem_like_path, variable_name=var_name, timestamp=timestamp)

def calculate_risk_index(flood_risk_map, mask):
    """
    Calcola un indice di rischio globale per la mappa di flood risk.
    Per ora usa una metrica semplice basata sulla media pesata delle aree allagate,
    considerando solo le zone terrestri (mask == 1).
    
    Args:
        flood_risk_map: numpy.ndarray con i valori di flood risk
        mask: numpy.ndarray con la maschera terra/mare (1 = terra, 0 = mare)
    
    Returns:
        float: indice di rischio normalizzato tra 0 e 1
    """
    # Considera solo le aree terrestri
    valid_area = flood_risk_map[mask == 1]
    
    if len(valid_area) == 0:
        return 0.0
    
    # Calcola statistiche sulle aree allagate
    mean_risk = np.mean(valid_area)
    max_risk = np.max(valid_area)
    flood_area_ratio = np.sum(valid_area > 0.5) / len(valid_area)  # percentuale di area con rischio > 0.5
    
    # Combina le metriche (puoi modificare questi pesi)
    risk_index = (0.4 * mean_risk + 0.3 * max_risk + 0.3 * flood_area_ratio)
    
    return float(risk_index)

def append_to_risk_log(timestamp, risk_index, runoff_stats, flood_stats, output_file):
    """
    Aggiunge una riga al file CSV del log di rischio con statistiche dettagliate.
    
    Args:
        timestamp: datetime object con il timestamp
        risk_index: float con l'indice di rischio
        runoff_stats: tuple (min, mean, max) delle statistiche del runoff
        flood_stats: tuple (min, mean, max) delle statistiche del flood risk
        output_file: str, path al file CSV di output
    """
    import csv
    import os
    
    # Crea l'header se il file non esiste
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'FloodRisk_Index', 
                           'Runoff_Min', 'Runoff_Mean', 'Runoff_Max',
                           'FloodRisk_Min', 'FloodRisk_Mean', 'FloodRisk_Max'])
        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M"),
            f"{risk_index:.4f}",
            f"{runoff_stats[0]:.4f}",
            f"{runoff_stats[1]:.4f}",
            f"{runoff_stats[2]:.4f}",
            f"{flood_stats[0]:.4f}",
            f"{flood_stats[1]:.4f}",
            f"{flood_stats[2]:.4f}"
        ])

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calcola il flood risk per un singolo file di previsione.")
    parser.add_argument("input_file", help="File di previsione (es: rdr0_d01_20250725Z1400_VMI_pred.tiff)")
    args = parser.parse_args()
    
    input_file = args.input_file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File di input non trovato: {input_file}")
        
    current_ts = parse_ts_from_fname(input_file)
    if current_ts is None:
        raise ValueError(f"Impossibile estrarre il timestamp dal nome file: {input_file}")
    
    print(colored(f"Processing file: {input_file}", "cyan"))
    print(colored(f"Timestamp: {current_ts.strftime('%Y-%m-%d %H:%M')}", "cyan"))

    # 1) DEM e mask mare
    print(colored("Loading DEM e sea mask...", "yellow"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # 2) Verifica D8
    print(colored("Checking D8 flow direction file...", "yellow"))
    if os.path.exists(config.D8_FILEPATH):
        print(colored(f"Using existing D8 file: {config.D8_FILEPATH}", "green"))
    else:
        raise FileNotFoundError(
            f"D8 file non trovato: {config.D8_FILEPATH}. Esegui prima il workflow completo per generarlo."
        )

    # 3) Allinea la CN alla griglia del DEM (una sola volta)
    print(colored("Processing Curve Number map...", "yellow"))
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # 4) Processo il file di input
    tag = ts_tag(current_ts)
    
    # 5) Controllo se esiste un file di flood risk precedente
    prev_flood_file = get_previous_flood_file(current_ts)
    previous_flood = load_previous_flood(prev_flood_file) if prev_flood_file else None
    
    print(colored("Elaboro il file con propagazione dello stato di piena...", "yellow"))
    print(colored(f"  -> process: {os.path.basename(input_file)}  (tag {tag})", "blue"))

    # Runoff istantaneo da precipitazione prevista
    runoff = process_radar_file(
        input_file,
        cn_map,
        MASK,
        config.DEM_FILEPATH
    )
    if runoff is None:
        raise ValueError(f"Impossibile calcolare il runoff per il file {input_file}")

    runoff_to_save = np.asarray(runoff, dtype=np.float32)
    np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Salva runoff per-step e ottieni le statistiche
    out_runoff = os.path.join(OUTPUT_DIR, f"runoff_{tag}.nc")
    print(colored(f"     saving runoff: {out_runoff}", "yellow"))
    runoff_stats = save_single_step(out_runoff, runoff_to_save, config.DEM_FILEPATH, var_name="runoff", timestamp=current_ts)

    # Flood istantaneo (dipende da runoff del passo)
    flood_inst = compute_flood_risk(config.D8_FILEPATH, runoff_to_save)

    # === Flood DINAMICO: usa stato precedente come condizione iniziale ===
    if previous_flood is None:
        print(colored("No previous flood state found, using instantaneous flood", "yellow"))
        flood_dyn = flood_inst
    else:
        print(colored("Combining with previous flood state", "green"))
        flood_dyn = np.maximum(flood_inst, previous_flood)

    # Salva il flood dinamico e ottieni le statistiche
    out_flood = os.path.join(OUTPUT_DIR, f"floodrisk_{tag}.nc")
    print(colored(f"Saving flood risk to {out_flood} ...", "yellow"))
    flood_stats = save_single_step(out_flood, flood_dyn.astype(np.float32), config.DEM_FILEPATH, var_name="floodrisk", timestamp=current_ts)
    
    # Calcola e salva l'indice di rischio e le statistiche
    risk_index = calculate_risk_index(flood_dyn, MASK)
    risk_log_file = os.path.join(OUTPUT_DIR, "flood_risk_index.csv")
    append_to_risk_log(current_ts, risk_index, runoff_stats, flood_stats, risk_log_file)
    print(colored(f"Risk index calculated and saved: {risk_index:.4f}", "cyan"))

    print(colored("Elaborazione completata ✅", "green"))

if __name__ == "__main__":
    main()