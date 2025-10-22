'''
import cdsapi
import zipfile
import os
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GLC_ZIP_FILE = os.path.join(DATA_DIR, "ee75a374499e6e965825d7f0a427407c.zip")
GLC_NETCDF_FILE = os.path.join(DATA_DIR, "data/C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.47.1153931748.18.4802470232.36.619987291.6.7499552751.nc")
GLC_TIFF_FILE = os.path.join(DATA_DIR, "glc_global.tif")

def download_glc():
    c = cdsapi.Client()
    c.retrieve(
        "satellite-land-cover",
        {
            "variable": "all",
            "year": ["2022"],
            "version": ["v2_1_1"],
            "area": [47.1153931748, 6.7499552751, 36.619987291, 18.4802470232]
        },
        GLC_ZIP_FILE
    )
    print("GLC scaricato con successo.")

def extract_netcdf(zip_file, destination_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        print("Contenuto del file ZIP:")
        print(zip_ref.namelist()) 
        
        zip_ref.extract('C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.41.5.15.8.39.9.13.7.nc', destination_dir)

    extracted_file = os.path.join(destination_dir, 'C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.41.5.15.8.39.9.13.7.nc')
    if os.path.exists(extracted_file):
        print(f"File {extracted_file} estratto correttamente!")
    else:
        print(f"Errore: il file {extracted_file} non è stato estratto.")

def convert_to_geotiff():
    ds = xr.open_dataset(GLC_NETCDF_FILE)
    variable = list(ds.data_vars.keys())[0]
    data = ds[variable].isel(time=0).values

    transform = from_bounds(ds.lon.min(), ds.lat.min(), ds.lon.max(), ds.lat.max(), data.shape[1], data.shape[0])

    with rasterio.open(
        GLC_TIFF_FILE,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    print("GLC convertito in GeoTIFF.")

def process_glc():
    download_glc()
    extract_netcdf(GLC_ZIP_FILE, DATA_DIR)
    convert_to_geotiff()
'''

import cdsapi
import zipfile
import os
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

# Cartella dei dati (../data rispetto a questo file)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GLC_ZIP_FILE = os.path.join(DATA_DIR, "ee75a374499e6e965825d7f0a427407c.zip")

# Bounding box italiano per l’area richiesta:
# ordine CDS API: [lat_max, lon_min, lat_min, lon_max]
ITALY_AREA = [47.1153931748, 6.7499552751, 36.619987291, 18.4802470232]

# Nome del file NetCDF atteso dopo il download (seguendo il pattern area-subset.north.east.south.west)
GLC_NETCDF_FILE = os.path.join(
    DATA_DIR,
    "C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.47.1153931748.18.4802470232.36.619987291.6.7499552751.nc"
)

# Nome del GeoTIFF generato per l’Italia
GLC_TIFF_FILE = os.path.join(DATA_DIR, "glc_italy.tif")

def download_glc():
    """Scarica il dataset di copertura del suolo C3S 2022 per l’Italia."""
    c = cdsapi.Client()
    c.retrieve(
        "satellite-land-cover",
        {
            "variable": "all",
            "year": ["2022"],
            "version": ["v2_1_1"],
            # area: [lat_max, lon_min, lat_min, lon_max]
            "area": ITALY_AREA,
        },
        GLC_ZIP_FILE,
    )
    print("GLC scaricato con successo.")

def extract_netcdf(zip_file, destination_dir):
    """
    Estrae dal file ZIP il NetCDF corrispondente all’Italia.
    Cerca il nome del file basandosi sulla costante GLC_NETCDF_FILE.
    """
    def extract_netcdf(zip_file, destination_dir):
        with zipfile.ZipFile(zip_file, "r") as z:
            nc_members = [m for m in z.namelist() if m.endswith(".nc")]
            if not nc_members:
                print(f"Nello ZIP non ci sono .nc. Contenuto: {z.namelist()}")
                return None
            target = nc_members[0]  # prendi il primo .nc
            z.extract(target, destination_dir)
            extracted = os.path.join(destination_dir, target)
            print(f"File estratto: {extracted}")
            return extracted


def convert_to_geotiff(nc_path):
    import numpy as np
    import xarray as xr
    import rasterio
    from rasterio.transform import from_bounds

    # 1) apri con h5netcdf e disattiva mask/scale per evitare bug in __getitem__
    ds = xr.open_dataset(
        nc_path,
        engine="h5netcdf",       # <-- chiave per evitare l'HDF error del backend netcdf4
        mask_and_scale=False,    # <-- niente _FillValue/scale durante il read
        decode_cf=False          # <-- evita decodifica automatica
    )

    # 2) prendi la prima variabile di dati
    var = list(ds.data_vars.keys())[0]
    da = ds[var]
    # gestisci la dimensione time se esiste
    if "time" in da.dims:
        da = da.isel(time=0)

    data = da.values

    # 3) bounds da coordinate
    lon_min = float(ds.lon.min())
    lon_max = float(ds.lon.max())
    lat_min = float(ds.lat.min())
    lat_max = float(ds.lat.max())

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max,
                            data.shape[1], data.shape[0])

    # 4) assicurati che il dtype sia adatto a GeoTIFF (uint8 qui va benissimo)
    if data.dtype.kind == "u" and data.dtype.itemsize == 1:
        out_dtype = data.dtype
    else:
        out_dtype = np.uint8
        data = data.astype(out_dtype, copy=False)

    with rasterio.open(
        GLC_TIFF_FILE, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=out_dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="deflate"
    ) as dst:
        dst.write(data, 1)

    ds.close()
    print("GLC convertito in GeoTIFF.")



def process_glc():
    download_glc()
    extracted_nc = extract_netcdf(GLC_ZIP_FILE, DATA_DIR)
    if not extracted_nc:
        raise RuntimeError("NetCDF non estratto dallo ZIP.")
    convert_to_geotiff(extracted_nc)

