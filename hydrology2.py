import os
import numpy as np
import rasterio
from rasterio import Affine
from tempfile import NamedTemporaryFile
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset

from raster_utils import sea_mask, align_radar_to_dem
from termcolor import colored


# =========================
#    RUNOFF (SCS-CN)
# =========================

def compute_runoff(precipitation, cn_map, mask):
    """
    Calcola il deflusso superficiale (runoff) con SCS-CN.
    precipitation: mm per step (array 2D)
    cn_map: Curve Number (array 2D)
    mask: 1=terra, 0=mare
    """
    # evita CN nulli o fuori terra
    cn_map = np.where((cn_map <= 0) | (mask == 0), 1e-6, cn_map).astype(np.float32)
    precipitation = np.where(mask == 0, 0.0, precipitation).astype(np.float32)

    # capacità di ritenzione potenziale
    S = np.maximum((1000.0 / cn_map) - 10.0, 0.0).astype(np.float32)

    runoff = np.zeros_like(precipitation, dtype=np.float32)
    valid = (precipitation > 0.2 * S) & (mask == 1)
    # classica formula SCS-CN
    num = (precipitation[valid] - 0.2 * S[valid]) ** 2
    den = precipitation[valid] + 0.8 * S[valid] + 1e-6
    runoff[valid] = (num / den).astype(np.float32)
    # fuori dalla terra = 0
    return np.where(mask == 1, runoff, 0.0).astype(np.float32)


def process_tiff(file_path, cn_map, mask):
    """
    Legge una precipitazione TIFF, corregge nodata/negativi e calcola il runoff.
    """
    with rasterio.open(file_path) as src:
        precipitation = src.read(1).astype(np.float32)
        nodata = src.nodata

    # gestisci nodata con tolleranza
    if nodata is not None:
        precipitation = np.where(np.isclose(precipitation, nodata, atol=1e-1), 0.0, precipitation)

    # clamp negativi
    precipitation = np.where(precipitation < 0.0, 0.0, precipitation).astype(np.float32)

    # calcola runoff
    runoff = compute_runoff(precipitation, cn_map, mask)
    # sanifica
    runoff = np.nan_to_num(runoff, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return runoff


def process_radar_file(file_path, cn_map, mask, dem_tiff):
    """
    Riallinea il file pioggia al DEM e calcola il runoff.
    """
    with NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        aligned = tmp.name

    try:
        align_radar_to_dem(file_path, dem_tiff, aligned)
        runoff = process_tiff(aligned, cn_map, mask)
    except Exception as e:
        print(colored(f"Errore processando {file_path}: {e}", "red"))
        runoff = None
    finally:
        try:
            os.remove(aligned)
        except Exception:
            pass

    return runoff


# =========================
#   FLOODING con STATO
# =========================

def precompute_d8_next(flow_direction_tiff):
    """
    Precalcola per ogni cella l'indice (i,j) della cella a valle secondo il codice D8.

    ATTENZIONE: mappa coerente con compute_d8() del tuo codice precedente:
      directions = [16, 8, 4, 2, 1, 32, 64, 128]
      shift      = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
    """
    with rasterio.open(flow_direction_tiff) as src:
        d8 = src.read(1).astype(np.uint16)

    H, W = d8.shape

    # Mappa codice -> offset (di, dj) coerente con la tua compute_d8
    code2off = {
        16: (0, -1),   # W
        8:  (1, -1),   # SW
        4:  (1,  0),   # S
        2:  (1,  1),   # SE
        1:  (0,  1),   # E
        32: (-1, 1),   # NE
        64: (-1, 0),   # N
        128:(-1,-1),   # NW
        0:  (0,  0),   # sink/piatto
    }

    # indici di base
    ii = np.arange(H)[:, None].repeat(W, axis=1)
    jj = np.arange(W)[None, :].repeat(H, axis=0)

    di = np.zeros_like(d8, dtype=np.int32)
    dj = np.zeros_like(d8, dtype=np.int32)
    for code, (oi, oj) in code2off.items():
        m = (d8 == code)
        di[m] = oi
        dj[m] = oj

    next_i = np.clip(ii + di, 0, H - 1)
    next_j = np.clip(jj + dj, 0, W - 1)
    return next_i, next_j


def route_once(H, next_i, next_j, r=0.8):
    """
    Spinge una frazione r di H verso la cella a valle e accumula.
    Ritorna il nuovo H (post-routing singolo).
    """
    H = H.astype(np.float32, copy=False)
    keep = (1.0 - r) * H
    to_push = (r * H).ravel()

    # destinazioni flatten
    flat_dest = (next_i.ravel() * H.shape[1] + next_j.ravel()).astype(np.int64)
    acc = np.bincount(flat_dest, weights=to_push, minlength=H.size).reshape(H.shape)

    return (keep + acc).astype(np.float32)


def step_flood(H, runoff_step, next_i, next_j, k=0.3, r=0.8, substeps=2, mask=None):
    """
    Aggiorna lo stato H (serbatoio per cella) usando:
      - perdita k
      - apporto runoff_step
      - routing D8 in substeps
    Ritorna (H_new, flood_map_step).
    """
    if mask is not None:
        runoff_step = np.where(mask == 1, runoff_step, 0.0).astype(np.float32)
        H = np.where(mask == 1, H, 0.0).astype(np.float32)

    # serbatoio con perdita
    H = (1.0 - k) * H + runoff_step
    # routing a piccoli passi
    for _ in range(int(substeps)):
        H = route_once(H, next_i, next_j, r=r)

    # una proxy ragionevole del "flood" allo step è l'H dopo routing
    flood = H.astype(np.float32)
    return H.astype(np.float32), flood.astype(np.float32)


# =========================
#       UTIL SALVATAGGIO
# =========================

def save_as_netcdf(data, output_path, reference_file):
    """
    Salva un array (2D) come NetCDF copiando georeferenziazione dal riferimento.
    Variabile unica chiamata 'runoff' o 'flood' a seconda del nome file.
    """
    data = np.asarray(data, dtype=np.float32)
    nrows, ncols = data.shape

    with rasterio.open(reference_file) as src:
        transform: Affine = src.transform

    # da trasformazione affine ricavo lon/lat (assumendo CRS geografico o quasi)
    xs = np.arange(ncols, dtype=np.float32)
    ys = np.arange(nrows, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    lon = transform.c + xx * transform.a + yy * transform.b
    lat = transform.f + xx * transform.d + yy * transform.e

    var_name = "runoff" if "runoff" in os.path.basename(output_path) else "flood"

    with Dataset(output_path, "w", format="NETCDF4") as nc:
        nc.createDimension("y", nrows)
        nc.createDimension("x", ncols)

        vlat = nc.createVariable("latitude", "f4", ("y", "x"))
        vlon = nc.createVariable("longitude", "f4", ("y", "x"))
        vlat.units = "degrees_north"
        vlon.units = "degrees_east"
        vlat[:] = lat.astype(np.float32)
        vlon[:] = lon.astype(np.float32)

        v = nc.createVariable(var_name, "f4", ("y", "x"))
        if var_name == "runoff":
            v.units = "mm"
            v.long_name = "surface runoff (per step)"
        else:
            v.units = "1"
            v.long_name = "flood proxy (state after routing)"

        v[:] = data

        nc.title = f"{var_name} map"
        nc.history = "created by hydrology.save_as_netcdf"
        nc.Conventions = "CF-1.8"
