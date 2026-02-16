#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 2025

@author: tsvetelina
"""

import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

"Need to download the cfgrib package as well to use as the engine to process the data"

# ================================
# Configuration
# ================================

# Working directory containing ERA5 GRIB files
grib_files = sorted(glob.glob('./cb9b983ba6d1bce207da78a06d7e402b.grib'))

nr_years = len(grib_files) # number of years to process

# ================================
# ERA5 variables should be split into two groups
# ================================

# 1. Surface / forecast variables (total precipitation, etc.)
#    These are usually dataType='fc' in ERA5, typeOfLevel='surface'
surface_vars = []  # You can add ["tp","cp","lsp"] etc. if needed

# 2. Wind variables at 100m
#    These are usually dataType='an' (analysis)
wind_vars = ["100u", "100v"]

# ================================
# Function to load ERA5 variables lazily with Dask
# ================================

def load_era5_variables(files, variables, level_type=None):
    """
    Load ERA5 GRIB variables using Dask.
    
    Parameters:
        files (list): List of GRIB file paths
        variables (list): List of variable shortNames to load
        level_type (str): 'surface' or 'heightAboveGround'
        level (int or None): specific level for heightAboveGround variables
    Returns:
        xarray.Dataset: concatenated dataset along time
    """
    datasets = []
    
    # Build filter_by_keys for cfgrib
    filter_keys = {"shortName": variables}
    if level_type is not None:
        filter_keys["typeOfLevel"] = level_type
    
    backend_kwargs = {"filter_by_keys": filter_keys}
    
    # Load each file lazily using Dask and print progress
    for f in files:
        print(f"Loading {f} for variables {variables} ...")
        ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs=backend_kwargs,
                             chunks={"time": 365})
        print(f"{f} loaded with variables: {list(ds.data_vars.keys())}")
        datasets.append(ds)
    
    # Concatenate along time and sort to ensure correct temporal order
    combined = xr.concat(datasets, dim="time").sortby("time")
    print(f"Combined dataset for variables {variables}:")
    print(combined)
    print("\n")
    
    return combined

# ================================
# Load surface and wind datasets
# ================================

# Load surface variables (tp, cp, etc.)
#ds_surface = load_era5_variables(grib_files, surface_vars, level_type="surface")

# Load wind variables at 100m
ds_wind = load_era5_variables(grib_files, wind_vars)



#%%
# ================================
# Compute wind speed and wind direction
# ================================
def compute_wind_speed(era5_u100, era5_v100):
    """
    Compute wind speed and return as xarray.DataArray preserving coordinates.
    """
    wspd = np.sqrt(era5_u100**2 + era5_v100**2)
    return xr.DataArray(
        wspd,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": era5_u100["time"],
            "step": era5_u100["step"],
            "latitude": era5_u100["latitude"],
            "longitude": era5_u100["longitude"],
            "number": era5_u100["number"],
            "surface": era5_u100["surface"],
            "valid_time": era5_u100["valid_time"],
        },
        name="wind_speed"
    )

def compute_wind_direction(era5_u100, era5_v100):
    """
    Compute wind direction (meteorological) and return as xarray.DataArray preserving coordinates.
    """
    # Meteorological convention: 0° = North, 90° = East
    wdir = (270 - np.arctan2(era5_u100, era5_v100) * 180 / np.pi) % 360
    return xr.DataArray(
        wdir,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": era5_u100["time"],
            "step": era5_u100["step"],
            "latitude": era5_u100["latitude"],
            "longitude": era5_u100["longitude"],
            "number": era5_u100["number"],
            "surface": era5_u100["surface"],
            "valid_time": era5_u100["valid_time"],
        },
        name="wind_direction"
    )

# ================================
# Plotting function
# ================================

def plot_map(data, title, units='', colormap='viridis', levels=None,
             vmin=None, vmax=None):
    """
    Plot a 2D xarray.DataArray on a Lambert Conformal map, save as PNG, and show.
    
    Parameters:
        data: xarray.DataArray with dims (latitude, longitude)
        title: str, plot title (also used for PNG filename)
        units: str, colorbar label
        colormap: str, matplotlib colormap
        levels: list or None, optional contour levels
        vmin: float or None, min colorbar value
        vmax: float or None, max colorbar value
    """
    # Lambert Conformal projection
    proj = ccrs.LambertConformal(
        central_longitude=2.5,
        central_latitude=53.6,
        standard_parallels=(30, 60)
    )
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})
    
    # Plot using pcolormesh
    img = data.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=colormap,
        add_colorbar=True,
        vmin=vmin,
        vmax=vmax,
        levels=levels
    )
    
    # Add coastlines and borders
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN)
    
    ax.set_title(title)
    
    # Colorbar label
    if img.colorbar:
        img.colorbar.set_label(units)
    
    # Save figure as PNG, 300 dpi, filename from title
    save_title = "plot-" + title.replace(" ", "_")
    plt.savefig(f"{save_title}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot as {save_title}.png")
    
    plt.show()






# ================================
# Computations and plotting
# ================================

# Compute time averages
print("Computing yearly-averaged accumulated precipitation...")
#tp_mean = ds_surface["tp"].sum(dim="time").sum(dim="step")/nr_years
print("Precipitation computed.")

print("Computing wind speed, wind direction...")
# Compute wind speed and wind direction from u, v components
ws = compute_wind_speed(ds_wind["u100"], ds_wind["v100"])
wd = compute_wind_direction(ds_wind["u100"], ds_wind["v100"])
print("Wind speed and wind direction computed.")

print("Computing time-averaged wind speed, wind direction...")
ws_mean = ws.mean(dim="time")
wd_mean = wd.mean(dim="time")
print("Wind components averaged.")


#%%
# ================================
# Show results
# ================================

# Precipitation
# Automatic continuous colorbar
"""plot_map(tp_mean*1000, title='Yearly-mean accumulated precipitation', 
         units='mm', colormap='Blues')
plot_map(tp_mean*1000, title='Yearly-mean accumulated precipitation (with vmin vmax)', 
         units='mm', colormap='Blues', vmin=200, vmax=1800)
# Discrete bins with levels
levels = [0, 250, 500, 750, 1000, 1250, 1500]
plot_map(tp_mean*1000, title='Yearly-mean accumulated precipitation (discrete bins)', 
         units='mm', colormap='Blues', levels=levels)"""
# multiplied by 1000 to convert from m to mm of precipitation
# Discrete bins with levels
levels = [0, 250, 500, 750, 1000, 1250, 1500]
# Wind speed
plot_map(ws_mean, title='Time-averaged wind speed at 100m', 
         units='m/s', vmin=1, vmax=11)

# Wind direction
levels_wdir = [0, 45, 90, 135, 180, 225, 270, 315, 360]
plot_map(wd_mean, title='Time-averaged wind direction at 100m', 
         units='deg', colormap='hsv',levels=levels_wdir)
