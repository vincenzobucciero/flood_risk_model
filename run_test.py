import os
import re
import glob
from collections import deque
from datetime import datetime, timedelta
import time
import sys

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


# --- util logging ---

def progress_bar(pct, width=24):
    pct = max(0.0, min(1.0, pct))
    done = int(pct * width)
    return "[" + "#" * done + "." * (width - done) + f"] {pct*100:5.1f}%"

def fmt_eta(seconds):
    if seconds < 60:
        return f"~{seconds:.0f}s"
    return f"~{seconds/60:.1f} min"


# --- Pipeline principale con stato + aggregato mobile 1h ---

def main():
    start_time_total = time.time()
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

    # Parametri flooding
    K = getattr(config, "FLOOD_DECAY_K", 0.3)
    R = getattr(config, "FLOOD_ROUTING_R", 0.8)
    SUBS = getattr(config, "FLOOD_SUBSTEPS", 2)
    print(colored(f"Flood params -> k={K}, r={R}, substeps={SUBS}", "yellow"))

    # 3) Allinea la CN alla griglia del DEM (una sola volta)
    print(colored("Processing Curve Number map...", "yellow"))
    align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)
    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)

    # 4) Finestra di lavoro (file predizioni)
    print(colored("Costruisco finestra di predizioni...", "yellow"))
    files_seq = build_expected_sequence(
        start_path=config.PREDICTION_START_FILE,
        steps=config.WINDOW_STEPS,
        step_minutes=config.STEP_MINUTES,
        directory=config.PREDICTION_DIR
    )

    if not files_seq:
        print(colored("Nessun file da processare. Controlla config e regex.", "red"))
        sys.exit(2)

    first_name = os.path.basename(files_seq[0])
    last_name = os.path.basename(files_seq[-1])
    print(colored(f"Selezionati {len(files_seq)} file", "yellow"))
    print(colored(f"  Primo: {first_name}", "yellow"))
    print(colored(f"  Ultimo: {last_name}", "yellow"))

    # 5) Loop sugli step con stato H e aggregato mobile 1h (6x10')
    print(colored("Calcolo runoff & flood con stato, + aggregato 1h mobile...", "yellow"))

    total_files = len(files_seq)
    H = None                               # stato di acqua per cella
    last6_flood = deque(maxlen=6)          # finestra mobile 1h (6 step da 10')
    last6_tags  = deque(maxlen=6)          # per naming 1h
    step_times = []                        # cronologia tempi step per ETA media

    for idx, pred_path in enumerate(files_seq, 1):
        step_start = time.time()
        ts = parse_ts_from_fname(pred_path)
        tag = ts_tag(ts)

        pct = idx / total_files
        print(colored(f"\n{progress_bar(pct)}  STEP {idx}/{total_files}  {os.path.basename(pred_path)}  (tag {tag})", "cyan"))

        # RUNOFF step
        runoff = process_radar_file(
            pred_path,
            cn_map,
            MASK,
            config.DEM_FILEPATH
        )
        if runoff is None:
            print(colored(f"     ⚠️ skip: runoff None per {pred_path}", "red"))
            continue

        # inizializza stato H
        if H is None:
            H = np.zeros_like(runoff, dtype=np.float32)

        # FLOOD con stato (serbatoio + D8 routing)
        H, flood = step_flood(
            H, runoff,
            next_i, next_j,
            k=K, r=R, substeps=SUBS,
            mask=MASK
        )

        # --- sanitizza prima del salvataggio ---
        runoff_to_save = np.asarray(runoff, dtype=np.float32)
        np.nan_to_num(runoff_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        flood_to_save = np.asarray(flood, dtype=np.float32)
        np.nan_to_num(flood_to_save, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Salva runoff e flood per step
        out_runoff = os.path.join(OUTPUT_DIR, f"runoff_{tag}.nc")
        out_flood  = os.path.join(OUTPUT_DIR, f"flood_{tag}.nc")

        try:
            print(colored(f"     saving step runoff → {out_runoff}", "yellow"))
            save_as_netcdf(runoff_to_save, out_runoff, config.DEM_FILEPATH)
            print(colored(f"     saving step flood  → {out_flood}", "yellow"))
            save_as_netcdf(flood_to_save, out_flood, config.DEM_FILEPATH)
        except Exception as e:
            print(colored(f"     ❌ errore salvataggio step {idx}: {e}", "red"))

        # Aggiornamento finestra mobile 1h
        last6_flood.append(flood_to_save)
        last6_tags.append(tag)

        if len(last6_flood) == 6:
            stack = np.stack(list(last6_flood), axis=0).astype(np.float32)
            flood_1h_sum = np.sum(stack, axis=0).astype(np.float32)
            flood_1h_max = np.max(stack, axis=0).astype(np.float32)

            # naming: 1h che termina all'istante corrente
            tag_start = last6_tags[0]
            tag_end   = last6_tags[-1]

            out_fsum = os.path.join(OUTPUT_DIR, f"flood_1h_sum_{tag_start}_to_{tag_end}.nc")
            out_fmax = os.path.join(OUTPUT_DIR, f"flood_1h_max_{tag_start}_to_{tag_end}.nc")

            try:
                print(colored(f"     saving 1h flood SUM → {out_fsum}", "yellow"))
                save_as_netcdf(flood_1h_sum, out_fsum, config.DEM_FILEPATH)
                print(colored(f"     saving 1h flood MAX → {out_fmax}", "yellow"))
                save_as_netcdf(flood_1h_max, out_fmax, config.DEM_FILEPATH)
            except Exception as e:
                print(colored(f"     ❌ errore salvataggio 1h agg.: {e}", "red"))

        # --- timing e ETA ---
        elapsed_step = time.time() - step_start
        step_times.append(elapsed_step)
        mean_step = np.mean(step_times[-5:])  # media ultimi 5 step (stima più stabile)
        remaining_est = (total_files - idx) * mean_step

        print(colored(f"     Step time: {elapsed_step:.1f}s | Avg(last5): {mean_step:.1f}s | ETA: {fmt_eta(remaining_est)}", "green"))

    elapsed_total = time.time() - start_time_total
    print(colored(f"\n✅ Completato in {elapsed_total/60:.1f} min totali ({elapsed_total:.0f}s).", "green"))
    print(colored(f"Output dir: {OUTPUT_DIR}", "yellow"))


if __name__ == "__main__":
    main()
