import xarray as xr

ds = xr.open_dataset("/storage/external_01/hiwefi/flood_outputs/flood_20250725Z1300.nc", chunks={"time": 1})
print(ds)

