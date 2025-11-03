import os
import re
import glob
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from termcolor import colored

from raster_utils import *
from hydrology2 import (
    process_radar_file,
    save_as_netcdf,
    precompute_d8_next,
    step_flood
)
import config


# === Dove salvare gli output (disco capiente) ===
OUTPUT_DIR = "/storage/external_01/hiwefi/flood_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Utilità per i file e i timestamp ---

FNAME_RE = re.compile(r".*?(\d{8})Z(\d{4}).*?_pred\.tif{1,2}f?$", re.IGNORECASE)
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
    """Ritorna tag 'YYYYMMDDZ%H%M' da un datetime (o 'unknown')."""
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

def build_sequence_after_start_by_position(start_path, steps, directory):
    """Restituisce 'steps' file: lo start + i successivi in ordine (senza richiedere timestamp esatti)."""
    all_preds = sorted(glob.glob(os.path.join(directory, "*_pred.tif*")))
    if not all_preds:
        raise FileNotFoundError(f"Nessun *_pred.tif* in {directory}")

    # prova a trovare start esatto nella lista
    if start_path in all_preds:
        idx0 = all_preds.index(start_path)
        seq = all_preds[idx0: idx0 + steps]
    else:
        # ricava ts dallo start e prendi il primo file con ts >= start
        start_ts = parse_ts_from_fname(start_path)
        if start_ts is None:
            raise ValueError(f"Impossibile ricavare ts da: {start_path}")
        def ts_or_min(p):
            ts = parse_ts_from_fname(p)
            return ts if ts is not None else datetime.min
        all_by_ts = sorted(all_preds, key=ts_or_min)
        idx0 = next((k for k,p in enumerate(all_by_ts) if (parse_ts_from_fname(p) or datetime.min) >= start_ts), 0)
        seq = all_by_ts[idx0: idx0 + steps]

    if len(seq) < steps:
        raise FileNotFoundError(f"Dopo lo start ho trovato solo {len(seq)} file; steps richiesti={steps}")
    return seq



# --- Pipeline principale con stato + aggregato mobile 1h ---

def main():
    print(colored(f"USING config from: {os.path.abspath(config.__file__)}", "yellow"))
    print(colored(f"PREDICTION_DIR      = {config.PREDICTION_DIR}", "yellow"))
    print(colored(f"PREDICTION_START    = {config.PREDICTION_START_FILE}", "yellow"))
    print(colored(f"WINDOW_STEPS        = {getattr(config, 'WINDOW_STEPS', None)}", "yellow"))
    print(colored(f"STEP_MINUTES        = {getattr(config, 'STEP_MINUTES', None)}", "yellow"))

    print(colored("Run flood: stato per step + aggregato 1h mobile", "cyan"))

    # 1) DEM e mask mare
    print(colored("Loading DEM e sea mask...", "yellow"))
    dem_data, dem_profile = latlon_load_and_plot_dem(config.DEM_FILEPATH)
    MASK = sea_mask(config.DEM_FILEPATH)

    # 2) Verifica D8 + precompute destinazioni
    print(colored("Checking D8 flow direction file...", "yellow"))
    if os.path.exists(config.D8_FILEPATH):
        print(colored(f"Using D8: {config.D8_FILEPATH}", "green"))
    else:
        raise FileNotFoundError(
            f"D8 file non trovato: {config.D8_FILEPATH}. Generarlo prima."
        )
    next_i, next_j = precompute_d8_next(config.D8_FILEPATH)

    # 3) Allinea la CN alla griglia del DEM (una sola volta)
    print(colored("Processing Curve Number map...", "yellow"))
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # 4) Finestra di lavoro (file predizioni)
    print(colored("Costruisco finestra di predizioni...", "yellow"))
    files_seq = build_sequence_after_start_by_position(
        start_path=config.PREDICTION_START_FILE,
        steps=config.WINDOW_STEPS,
        directory=config.PREDICTION_DIR
    )
    cand = sorted(glob.glob(os.path.join(config.PREDICTION_DIR, "*_pred.tif*")))
    print(colored(f"Candidati trovati: {len(cand)}", "yellow"))
    for p in cand[:12]:
        print("   -", os.path.basename(p), "->", parse_ts_from_fname(p))

    print(colored(f"Finestra selezionata ({len(files_seq)}):", "yellow"))
    for i, f in enumerate(files_seq, 1):
        print(f"  [{i}] {os.path.basename(f)}  -> {parse_ts_from_fname(f)}")

    for i, f in enumerate(files_seq, 1):
        print(colored(f"  [{i}/{len(files_seq)}] {os.path.basename(f)}", "blue"))

    # 5) Loop sugli step con stato H e aggregato mobile 1h (6x10')
    print(colored("Calcolo runoff & flood con stato, + aggregato 1h mobile...", "yellow"))

    H = None                               # stato di acqua per cella
    last6_flood = deque(maxlen=6)          # finestra mobile 1h (6 step da 10')
    last6_tags  = deque(maxlen=6)          # per naming 1h

    for idx, pred_path in enumerate(files_seq, 1):
        ts = parse_ts_from_fname(pred_path)
        tag = ts_tag(ts)
        print(colored(f"  -> step {idx}: {os.path.basename(pred_path)}  (tag {tag})", "blue"))

        # RUNOFF step
        runoff = process_radar_file(
            pred_path,
            cn_map,
            MASK,
            config.DEM_FILEPATH
        )
        if runoff is None:
            print(colored(f"     skip: runoff None per {pred_path}", "red"))
            continue

        # inizializza stato H
        if H is None:
            H = np.zeros_like(runoff, dtype=np.float32)

        # FLOOD con stato (serbatoio + D8 routing)
        H, flood = step_flood(
            H, runoff,
            next_i, next_j,
            k=getattr(config, "FLOOD_DECAY_K", 0.3),
            r=getattr(config, "FLOOD_ROUTING_R", 0.8),
            substeps=getattr(config, "FLOOD_SUBSTEPS", 2),
            mask=MASK
        )

        # --- sanitizza prima del salvataggio ---
        runoff_to_save = np.asarray(runoff, dtype=np.float32)
        np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        flood_to_save = np.asarray(flood, dtype=np.float32)
        np.nan_to_num(flood_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Salva runoff e flood per step
        out_runoff = os.path.join(OUTPUT_DIR, f"runoff_{tag}.nc")
        print(colored(f"     saving step runoff: {out_runoff}", "yellow"))
        save_as_netcdf(runoff_to_save, out_runoff, config.DEM_FILEPATH)

        out_flood = os.path.join(OUTPUT_DIR, f"flood_{tag}.nc")
        print(colored(f"     saving step flood:  {out_flood}", "yellow"))
        save_as_netcdf(flood_to_save, out_flood, config.DEM_FILEPATH)

        # Aggiornamento finestra mobile 1h
        last6_flood.append(flood_to_save)
        last6_tags.append(tag)

        if len(last6_flood) == 6:
            stack = np.stack(list(last6_flood), axis=0).astype(np.float32)
            flood_1h_sum = np.sum(stack, axis=0).astype(np.float32)
            flood_1h_max = np.max(stack, axis=0).astype(np.float32)

            # naming: 1h che termina all'istante corrente
            tag_start = last6_tags[0]
            tag_end = last6_tags[-1]

            out_fsum = os.path.join(OUTPUT_DIR, f"flood_1h_sum_{tag_start}_to_{tag_end}.nc")
            out_fmax = os.path.join(OUTPUT_DIR, f"flood_1h_max_{tag_start}_to_{tag_end}.nc")

            print(colored(f"     saving 1h flood SUM: {out_fsum}", "yellow"))
            save_as_netcdf(flood_1h_sum, out_fsum, config.DEM_FILEPATH)

            print(colored(f"     saving 1h flood MAX: {out_fmax}", "yellow"))
            save_as_netcdf(flood_1h_max, out_fmax, config.DEM_FILEPATH)

    print(colored("Flood per step + aggregati 1h completati ✅", "green"))


if __name__ == "__main__":
    main()
