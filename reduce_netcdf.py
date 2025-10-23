import xarray as xr
import numpy as np
from pathlib import Path
import os

def reduce_netcdf(input_path, output_path, reduction_factor=10):
    """
    Reduce the size of a NetCDF file by downsampling the data.
    
    Args:
        input_path: Path to the input NetCDF file
        output_path: Path to save the reduced NetCDF file
        reduction_factor: Factor by which to reduce the resolution
    """
    print(f"Loading NetCDF file: {input_path}")
    ds = xr.open_dataset(input_path)
    
    # Get original dimensions
    orig_shape = ds.runoff.shape
    print(f"\nOriginal dimensions: {orig_shape}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024*1024*1024):.2f} GB")
    
    # Calculate new dimensions ensuring they're divisible by reduction_factor
    new_y = (orig_shape[0] // reduction_factor) * reduction_factor
    new_x = (orig_shape[1] // reduction_factor) * reduction_factor
    
    # Trim the arrays to make them divisible by reduction_factor
    ds_trimmed = ds.isel(y=slice(0, new_y), x=slice(0, new_x))
    
    # Reduce resolution
    ds_reduced = ds_trimmed.coarsen(y=reduction_factor, x=reduction_factor, boundary='trim').mean()
    
    print(f"\nNew dimensions: {ds_reduced.runoff.shape}")
    
    # Save reduced dataset
    print(f"\nSaving reduced NetCDF to: {output_path}")
    ds_reduced.to_netcdf(output_path)
    
    # Print file size comparison
    new_size = os.path.getsize(output_path) / (1024*1024*1024)
    print(f"New file size: {new_size:.2f} GB")
    print(f"Size reduction: {(1 - new_size/(os.path.getsize(input_path)/(1024*1024*1024))) * 100:.1f}%")
    
    # Print value statistics
    print("\nValue statistics:")
    print(f"Original - Min: {float(ds.runoff.min()):.2f}, Max: {float(ds.runoff.max()):.2f}, Mean: {float(ds.runoff.mean()):.2f}")
    print(f"Reduced  - Min: {float(ds_reduced.runoff.min()):.2f}, Max: {float(ds_reduced.runoff.max()):.2f}, Mean: {float(ds_reduced.runoff.mean()):.2f}")
    
    return ds_reduced

if __name__ == "__main__":
    input_file = "outputs/runoff_single.nc"
    output_file = "outputs/runoff_single_reduced.nc"
    
    # Create a reduced version with 1/10th resolution
    ds_reduced = reduce_netcdf(input_file, output_file, reduction_factor=10)
    print("\nReduction complete! You can now transfer the reduced file to your local machine.")