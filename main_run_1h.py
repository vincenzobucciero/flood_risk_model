'''
import os
import re
import glob
from datetime import datetime, timedelta

import numpy as np
from termcolor import colored

from raster_utils import *
from hydrology import *
import config

# --- Utilità per i file e i timestamp ---

FNAME_RE = re.compile(r".*?(\d{8})Z(\d{4})_VMI.*?_pred\.tif{1,2}f?$")
# Esempio: rdr0_d01_20251008Z0300_VMI_pred.tiff
# Gruppi: 20251008 e 0300

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

# --- Pipeline principale 1h ---

def main():
    print(colored("Test run 1h: somma di 6 predizioni runoff (10' ciascuna)", "cyan"))
    os.makedirs("outputs", exist_ok=True)

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

    # 4) Lista dei 6 file da sommare
    print(colored("Costruisco la finestra 1h di predizioni...", "yellow"))
    files_1h = build_expected_sequence(
        start_path=config.PREDICTION_START_FILE,
        steps=config.WINDOW_STEPS,
        step_minutes=config.STEP_MINUTES,
        directory=config.PREDICTION_DIR
    )
    for i, f in enumerate(files_1h, 1):
        print(colored(f"  [{i}/{len(files_1h)}] {os.path.basename(f)}", "blue"))

    # 5) Calcolo, salvataggio singoli (runoff+flood), accumulo
    print(colored("Calcolo runoff & flood risk per step e sommo su 1h...", "yellow"))
    runoff_sum = None

    for idx, pred_path in enumerate(files_1h, 1):
        ts = parse_ts_from_fname(pred_path)
        tag = ts_tag(ts)
        print(colored(f"  -> process {idx}: {os.path.basename(pred_path)}  (tag {tag})", "blue"))

        runoff = process_radar_file(
            pred_path,
            cn_map,
            MASK,
            config.DEM_FILEPATH
        )
        if runoff is None:
            print(colored(f"     skip: runoff None per {pred_path}", "red"))
            continue

        # --- sanitizza prima del salvataggio ---
        runoff_to_save = np.asarray(runoff, dtype=np.float32)
        np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Salva runoff singolo
        out_single = f"outputs/runoff_{tag}.nc"
        print(colored(f"     saving single-step runoff: {out_single}", "yellow"))
        save_as_netcdf(runoff_to_save, out_single, config.DEM_FILEPATH)

        # Flood risk per step
        flood = compute_flood_risk(config.D8_FILEPATH, runoff_to_save)
        flood_to_save = np.asarray(np.nan_to_num(flood, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
        out_flood = f"outputs/floodrisk_{tag}.nc"
        print(colored(f"     saving single-step flood risk: {out_flood}", "yellow"))
        save_as_netcdf(flood_to_save, out_flood, config.DEM_FILEPATH)

        # Accumula (somma ignorando NaN)
        if runoff_sum is None:
            runoff_sum = np.zeros_like(runoff_to_save, dtype=np.float32)
        valid = np.isfinite(runoff_to_save)
        runoff_sum[valid] += runoff_to_save[valid]

    # 6) Salvataggi 1h (runoff + flood)
    if runoff_sum is None:
        raise RuntimeError("Nessun file valido processato: impossibile produrre la somma 1h.")

    start_ts = parse_ts_from_fname(files_1h[0])
    tag_start = ts_tag(start_ts)

    # runoff 1h
    out_nc = f"outputs/runoff_1h_{tag_start}.nc"
    print(colored(f"Saving 1h runoff to {out_nc} ...", "yellow"))
    runoff_sum_to_save = np.asarray(runoff_sum, dtype=np.float32)
    np.nan_to_num(runoff_sum_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    save_as_netcdf(runoff_sum_to_save, out_nc, config.DEM_FILEPATH)

    # flood risk 1h (calcolato sul runoff 1h)
    flood_1h = compute_flood_risk(config.D8_FILEPATH, runoff_sum_to_save)
    flood_1h_to_save = np.asarray(np.nan_to_num(flood_1h, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
    out_flood_1h = f"outputs/floodrisk_1h_{tag_start}.nc"
    print(colored(f"Saving 1h flood risk to {out_flood_1h} ...", "yellow"))
    save_as_netcdf(flood_1h_to_save, out_flood_1h, config.DEM_FILEPATH)

    print(colored("Runoff & Flood risk 1h completati ✅", "green"))

if __name__ == "__main__":
    main()
'''


import os
import re
import glob
from datetime import datetime, timedelta

import numpy as np
from termcolor import colored

from raster_utils import *
from hydrology import *
import config

# === Dove salvare gli output (disco capiente) ===
OUTPUT_DIR = "/storage/external_01/hiwefi/flood_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utilità per i file e i timestamp ---

FNAME_RE = re.compile(r".*?(\d{8})Z(\d{4})_VMI.*?_pred\.tif{1,2}f?$")
# Esempio: rdr0_d01_20251008Z0300_VMI_pred.tiff
# Gruppi: 20251008 e 0300

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

# --- Pipeline principale 1h ---

def main():
    print(colored("Test run 1h: somma di 6 predizioni runoff (10' ciascuna)", "cyan"))

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

    # 4) Lista dei 6 file da sommare
    print(colored("Costruisco la finestra 1h di predizioni...", "yellow"))
    files_1h = build_expected_sequence(
        start_path=config.PREDICTION_START_FILE,
        steps=config.WINDOW_STEPS,
        step_minutes=config.STEP_MINUTES,
        directory=config.PREDICTION_DIR
    )
    for i, f in enumerate(files_1h, 1):
        print(colored(f"  [{i}/{len(files_1h)}] {os.path.basename(f)}", "blue"))

    # 5) Calcolo, salvataggio singoli (runoff+flood), accumulo
    print(colored("Calcolo runoff & flood risk per step e sommo su 1h...", "yellow"))
    runoff_sum = None

    for idx, pred_path in enumerate(files_1h, 1):
        ts = parse_ts_from_fname(pred_path)
        tag = ts_tag(ts)
        print(colored(f"  -> process {idx}: {os.path.basename(pred_path)}  (tag {tag})", "blue"))

        runoff = process_radar_file(
            pred_path,
            cn_map,
            MASK,
            config.DEM_FILEPATH
        )
        if runoff is None:
            print(colored(f"     skip: runoff None per {pred_path}", "red"))
            continue

        # --- sanitizza prima del salvataggio ---
        runoff_to_save = np.asarray(runoff, dtype=np.float32)
        np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Salva runoff singolo
        out_single = os.path.join(OUTPUT_DIR, f"runoff_{tag}.nc")
        print(colored(f"     saving single-step runoff: {out_single}", "yellow"))
        save_as_netcdf(runoff_to_save, out_single, config.DEM_FILEPATH)

        # Flood risk per step
        flood = compute_flood_risk(config.D8_FILEPATH, runoff_to_save)
        flood_to_save = np.asarray(np.nan_to_num(flood, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
        out_flood = os.path.join(OUTPUT_DIR, f"floodrisk_{tag}.nc")
        print(colored(f"     saving single-step flood risk: {out_flood}", "yellow"))
        save_as_netcdf(flood_to_save, out_flood, config.DEM_FILEPATH)

        # Accumula (somma ignorando NaN)
        if runoff_sum is None:
            runoff_sum = np.zeros_like(runoff_to_save, dtype=np.float32)
        valid = np.isfinite(runoff_to_save)
        runoff_sum[valid] += runoff_to_save[valid]

    # 6) Salvataggi 1h (runoff + flood)
    if runoff_sum is None:
        raise RuntimeError("Nessun file valido processato: impossibile produrre la somma 1h.")

    start_ts = parse_ts_from_fname(files_1h[0])
    tag_start = ts_tag(start_ts)

    # runoff 1h
    out_nc = os.path.join(OUTPUT_DIR, f"runoff_1h_{tag_start}.nc")
    print(colored(f"Saving 1h runoff to {out_nc} ...", "yellow"))
    runoff_sum_to_save = np.asarray(runoff_sum, dtype=np.float32)
    np.nan_to_num(runoff_sum_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    save_as_netcdf(runoff_sum_to_save, out_nc, config.DEM_FILEPATH)

    # flood risk 1h (calcolato sul runoff 1h)
    flood_1h = compute_flood_risk(config.D8_FILEPATH, runoff_sum_to_save)
    flood_1h_to_save = np.asarray(np.nan_to_num(flood_1h, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
    out_flood_1h = os.path.join(OUTPUT_DIR, f"floodrisk_1h_{tag_start}.nc")
    print(colored(f"Saving 1h flood risk to {out_flood_1h} ...", "yellow"))
    save_as_netcdf(flood_1h_to_save, out_flood_1h, config.DEM_FILEPATH)

    print(colored("Runoff & Flood risk 1h completati ✅", "green"))

if __name__ == "__main__":
    main()
