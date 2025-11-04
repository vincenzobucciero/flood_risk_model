# main_run_1h.py
import os
import re
import glob
from datetime import datetime, timedelta

import numpy as np
from termcolor import colored

from raster_utils import *
from hydrology2 import *
import xarray as xr
import config

# === Dove salvare gli output per-step (compatibile con i tuoi risultati attuali) ===
OUTPUT_DIR = "/storage/external_01/hiwefi/flood_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        ds = xr.open_dataset(file_path)
        flood = ds["flood"].values if "flood" in ds else ds["value"].values
        ds.close()
        return flood
    except Exception as e:
        print(colored(f"Error loading previous flood risk: {e}", "red"))
        return None

# === Radice per la struttura prf2/d02/archive/... del file unico ===
ARCHIVE_ROOT = getattr(config, "ARCHIVE_ROOT", "/storage/external_01/hiwefi")

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

def build_expected_sequence(start_path, steps, step_minutes, directory):
    """
    Dato un file di partenza, costruisce la sequenza attesa di 'steps' file
    a distanza di 'step_minutes'. Se alcuni mancano, usa i successivi ordinati.
    """
    start_ts = parse_ts_from_fname(start_path)
    if start_ts is None:
        raise ValueError(f"Impossibile leggere timestamp da: {start_path}")

    all_preds = sorted(
        glob.glob(os.path.join(directory, "*_pred.tif*")),
        key=lambda p: (parse_ts_from_fname(p) or datetime.min)
    )
    by_ts = {parse_ts_from_fname(p): p for p in all_preds if parse_ts_from_fname(p) is not None}

    expected, missing = [], []
    for i in range(steps):
        ts_i = start_ts + timedelta(minutes=step_minutes * i)
        path_i = by_ts.get(ts_i)
        if path_i and os.path.exists(path_i):
            expected.append(path_i)
        else:
            missing.append(ts_i)

    if not missing:
        return expected

    # Fallback: prendi i successivi (incluso start) ordinati per ts
    seq = [p for p in all_preds if parse_ts_from_fname(p) and parse_ts_from_fname(p) >= start_ts]
    if len(seq) >= steps:
        print(colored(f"Attenzione: mancavano {len(missing)} file attesi; uso fallback sequenziale.", "red"))
        return seq[:steps]

    missing_str = ", ".join(ts.strftime("%Y-%m-%d %H:%M") for ts in missing)
    raise FileNotFoundError(
        f"Non riesco a comporre {steps} file consecutivi da {start_path}. Mancanti: {missing_str}."
    )

def ensure_prf2_archive_dir(dt_start):
    """Crea la directory prf2/d02/archive/YYYY/MM/DD sotto ARCHIVE_ROOT."""
    yyyy = dt_start.strftime("%Y")
    mm = dt_start.strftime("%m")
    dd = dt_start.strftime("%d")
    dest_dir = os.path.join(ARCHIVE_ROOT, "prf2", "d02", "archive", yyyy, mm, dd)
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir

def save_single_step(path, arr, dem_like_path, var_name="value"):
    """Salva un array 2D come NetCDF con georeferenziazione ricavata dal dem_like_path."""
    save_as_netcdf(arr.astype(np.float32), path, dem_like_path)

def save_hourly_prf2(flood_stack, hourly_risk, times, dem_like_path, dt_start):
    """
    Salva un unico NetCDF con:
      - flood(time, y, x): le 6 mappe di flooding DINAMICO
      - hourly_risk(y, x): fattore di rischio orario
      - time: 6 timestamp dei passi da 10'
    Nomenclatura/struttura: prf2/d02/archive/YYYY/MM/DD/prf2_d02_YYYYMMDDZHH00.nc
    """
    # Stack a (time,y,x)
    flood_stack = np.asarray(flood_stack, dtype=np.float32)
    # Timestamp come np.datetime64
    time_vals = np.array([np.datetime64(t) for t in times])

    # Costruisci dataset
    ds = xr.Dataset(
        data_vars={
            "flood": (("time", "y", "x"), flood_stack),
            "hourly_risk": (("y", "x"), hourly_risk.astype(np.float32)),
        },
        coords={"time": time_vals},
        attrs={"title": "Nowcast Flood - PRF2 archive", "convention": "simple"},
    )

    # Percorso di destinazione
    dest_dir = ensure_prf2_archive_dir(dt_start)
    # Nome file: timestamp arrotondato all’ora “di inizio finestra”
    # Esempio: per start 2025-07-25 13:00 -> prf2_d02_20250725Z1300.nc
    tag_start = dt_start.strftime("%Y%m%dZ%H00")
    dest_path = os.path.join(dest_dir, f"prf2_d02_{tag_start}.nc")

    # Scrivi NetCDF
    comp = dict(zlib=True, complevel=4, dtype="float32", shuffle=True)
    ds["flood"].encoding = comp
    ds["hourly_risk"].encoding = comp
    ds.to_netcdf(dest_path)

    return dest_path

# --- Pipeline principale 1h ---

def main():
    print(colored("Nowcast 1h: 6 predizioni da 10' con flooding DINAMICO (stato di piena propagato)", "cyan"))

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

    # 3) Allinea la CN alla griglia del DEM (una sola volta)
    print(colored("Processing Curve Number map...", "yellow"))
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # 4) Lista dei 6 file attesi (o fallback ordinato)
    print(colored("Costruisco la finestra 1h di predizioni (6×10')...", "yellow"))
    files_1h = build_expected_sequence(
        start_path=config.PREDICTION_START_FILE,
        steps=config.WINDOW_STEPS,
        step_minutes=config.STEP_MINUTES,
        directory=config.PREDICTION_DIR
    )
    for i, f in enumerate(files_1h, 1):
        print(colored(f"  [{i}/{len(files_1h)}] {os.path.basename(f)}", "blue"))

    # 5) Calcolo (runoff + flood dinamico) e salvataggi per-step
    print(colored("Elaboro i 6 step con propagazione dello stato di piena...", "yellow"))
    runoff_sum = None
    previous_flood = None             # <-- stato di piena che si propaga
    flood_dynamic_stack = []          # per costruire il file unico PRF2
    times = []

    for idx, pred_path in enumerate(files_1h, 1):
        ts = parse_ts_from_fname(pred_path)
        tag = ts_tag(ts)
        times.append(ts)
        print(colored(f"  -> process {idx}: {os.path.basename(pred_path)}  (tag {tag})", "blue"))

        # Per il primo file, controlla se esiste un flood precedente
        if idx == 1:
            prev_flood_file = get_previous_flood_file(ts)
            if prev_flood_file:
                previous_flood = load_previous_flood(prev_flood_file)
                print(colored(f"Loaded previous flood state from: {prev_flood_file}", "green"))

        # Runoff istantaneo da precipitazione prevista
        runoff = process_radar_file(
            pred_path,
            cn_map,
            MASK,
            config.DEM_FILEPATH
        )
        if runoff is None:
            print(colored(f"     skip: runoff None per {pred_path}", "red"))
            continue

        runoff_to_save = np.asarray(runoff, dtype=np.float32)
        np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Salva runoff per-step (compatibilità con i tuoi output attuali)
        out_runoff = os.path.join(OUTPUT_DIR, f"runoff_{tag}.nc")
        print(colored(f"     saving single-step runoff: {out_runoff}", "yellow"))
        save_single_step(out_runoff, runoff_to_save, config.DEM_FILEPATH, var_name="runoff")

        # Flood istantaneo (dipende da runoff del passo)
        flood_inst = compute_flood_risk(config.D8_FILEPATH, runoff_to_save)

        # === Flood DINAMICO: usa stato precedente come condizione iniziale ===
        if previous_flood is None:
            flood_dyn = flood_inst
        else:
            # Prendi il massimo tra lo stato precedente e quello attuale
            flood_dyn = np.maximum(flood_inst, previous_flood)

        previous_flood = flood_dyn.copy()

        # Salva il flood dinamico per-step (sovrascriviamo il nome floodrisk_*)
        out_flood = os.path.join(OUTPUT_DIR, f"floodrisk_{tag}.nc")
        print(colored(f"     saving single-step FLOOD (DINAMICO): {out_flood}", "yellow"))
        save_single_step(out_flood, flood_dyn.astype(np.float32), config.DEM_FILEPATH, var_name="flood")

        # Accumula runoff su 1h
        if runoff_sum is None:
            runoff_sum = np.zeros_like(runoff_to_save)
        valid = np.isfinite(runoff_to_save)
        runoff_sum[valid] += runoff_to_save[valid]

        # Mantieni lo stack per il file unico PRF2
        flood_dynamic_stack.append(flood_dyn.astype(np.float32))

    # 6) Salvataggi 1h (runoff cumulato + rischio orario)
    if runoff_sum is None:
        raise ValueError("Nessun runoff valido elaborato.")

    start_ts = parse_ts_from_fname(files_1h[0])
    tag_start = ts_tag(start_ts)

    # Runoff 1h
    out_nc_runoff1h = os.path.join(OUTPUT_DIR, f"runoff_1h_{tag_start}.nc")
    print(colored(f"Saving 1h runoff to {out_nc_runoff1h} ...", "yellow"))
    runoff_sum_to_save = np.asarray(runoff_sum, dtype=np.float32)
    np.nan_to_num(runoff_sum_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    save_single_step(out_nc_runoff1h, runoff_sum_to_save, config.DEM_FILEPATH, var_name="runoff_1h")

    # Fattore di rischio orario (proxy: compute_flood_risk sul runoff cumulato)
    flood_1h = compute_flood_risk(config.D8_FILEPATH, runoff_sum_to_save)
    flood_1h = np.asarray(np.nan_to_num(flood_1h, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

    out_nc_flood1h = os.path.join(OUTPUT_DIR, f"floodrisk_1h_{tag_start}.nc")
    print(colored(f"Saving 1h flood risk to {out_nc_flood1h} ...", "yellow"))
    save_single_step(out_nc_flood1h, flood_1h, config.DEM_FILEPATH, var_name="floodrisk_1h")

    # 7) File unico in stile PRF2 con 6 mappe flood dinamiche + rischio orario
    print(colored("Costruisco il NetCDF unico PRF2 con le 6 mappe dinamiche + rischio orario...", "yellow"))
    prf2_path = save_hourly_prf2(
        flood_stack=flood_dynamic_stack,
        hourly_risk=flood_1h,
        times=times,
        dem_like_path=config.DEM_FILEPATH,
        dt_start=start_ts
    )
    print(colored(f"File PRF2 scritto in: {prf2_path}", "green"))

    print(colored("Runoff & Flood DINAMICO 1h completati ✅", "green"))

if __name__ == "__main__":
    main()
