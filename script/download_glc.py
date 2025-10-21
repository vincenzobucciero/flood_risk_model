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
GLC_ZIP_FILE = os.path.join(DATA_DIR, "glc_global.zip")

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
    target_file = os.path.basename(GLC_NETCDF_FILE)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        if target_file not in zip_ref.namelist():
            print(f"File {target_file} non trovato nello ZIP; contenuto: {zip_ref.namelist()}")
            return
        zip_ref.extract(target_file, destination_dir)

    extracted_file = os.path.join(destination_dir, target_file)
    if os.path.exists(extracted_file):
        print(f"File {extracted_file} estratto correttamente!")
    else:
        print(f"Errore: il file {extracted_file} non è stato estratto.")

def convert_to_geotiff():
    """Converte il NetCDF estratto in un GeoTIFF (EPSG:4326)."""
    ds = xr.open_dataset(GLC_NETCDF_FILE)
    # Assumi che la variabile dati sia la prima nel NetCDF
    variable = list(ds.data_vars.keys())[0]
    data = ds[variable].isel(time=0).values

    # Calcola il transform dai bounds del dataset
    transform = from_bounds(
        float(ds.lon.min()), float(ds.lat.min()),
        float(ds.lon.max()), float(ds.lat.max()),
        data.shape[1], data.shape[0]
    )

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
    """Scarica, estrae e converte il dataset di copertura del suolo per l’Italia."""
    download_glc()
    extract_netcdf(GLC_ZIP_FILE, DATA_DIR)
    convert_to_geotiff()
