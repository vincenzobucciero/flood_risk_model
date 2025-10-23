#!/usr/bin/env python3
"""Process all prediction TIFFs in the configured prediction directory.

Creates one NetCDF per input TIFF in the outputs/ directory using the same
processing flow as `main_run_local.py` but file-by-file.
"""
import os
from pathlib import Path
import config
from raster_utils import latlon_load_and_plot_land_cover, sea_mask
from hydrology import process_radar_file, save_as_netcdf


def main():
    pred_dir = Path(config.PREDICTION_DIR)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Load CN map and mask
    if not Path(config.ALIGNED_CN_FILEPATH).exists():
        from raster_utils import align_radar_to_dem
        align_radar_to_dem(config.CN_MAP_FILEPATH, config.DEM_FILEPATH, config.ALIGNED_CN_FILEPATH)

    cn_map, _ = latlon_load_and_plot_land_cover(config.ALIGNED_CN_FILEPATH)
    mask = sea_mask(config.DEM_FILEPATH)

    tiffs = sorted([p for p in pred_dir.glob('*.tiff')])
    if not tiffs:
        print('No prediction TIFFs found in', pred_dir)
        return

    for t in tiffs:
        print('Processing', t.name)
        runoff = process_radar_file(str(t), cn_map, mask, config.DEM_FILEPATH)
        if runoff is None:
            print('Failed to compute runoff for', t.name)
            continue

        out_path = out_dir / (t.stem + '.nc')
        save_as_netcdf(runoff, str(out_path), config.DEM_FILEPATH)
        print('Saved', out_path)


if __name__ == '__main__':
    main()
