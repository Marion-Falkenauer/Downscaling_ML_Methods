#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/12/25

Pre-processing to get the .grib files into Panda dataframes : @author: tsvetelina

Rest of the code : @author: Marion Falkenauer (with the help of Copilot for the code's syntax)

This script loads ERA5 GRIB files, extracts wind variables at 100m,
and visualizes them on a map. It also includes functions to compute wind speed and direction (if want to use those instead of u/v components).

Then, the pre-processed data is prepared for machine learning by creating TensorFlow datasets for high-resolution (HR) and low-resolution (LR) wind data.
In this code, the LR is here obtained through simple downsampling by taking every RF-th pixel to create the LR dataset, from the Reduction Factor RF.

Those two datasets are then used to be fed thourgh a simple SRCNN, whose architecture is taken from the literature.
The post-processing for ML results (learning curve monitoring for overfitting etc, visualisation of HR/ LR /ML results) are also included.

This code is thus a strong basis for the second step of the research project.

"""

import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import tensorflow as tf
from matplotlib import animation
from tensorflow import keras
#from tensorflow.keras import layers   # Suggested here
from keras import layers # Suggestion to use keras directly
from sklearn.preprocessing import MinMaxScaler

# For saving the plots as PNG files
import os
# For saving the history of the test metrics in a CSV file;

import csv
from datetime import datetime
import pickle
import pandas as pd
import time # Module for counter
import psutil

plt.rcParams.update({
    "font.size": 25,           # default text size
    "axes.labelsize": 20,      # x and y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "axes.titlesize": 20
})


# ================================
# Configuration
# ================================
# Base directory containing everything
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to a file in the same directory
ERA5_files = os.path.join(BASE_DIR, "Data/data.grib")
CERRA_files = os.path.join(BASE_DIR, "Data/CERRA/CERRA_100m_3h_2024_full_domain.nc")
# Working directory containing ERA5 GRIB files
# Line for the yearly prediction
grib_files = sorted(glob.glob(ERA5_files))
# Line for the toy dataset (112 samples)
#grib_files = sorted(glob.glob(r'.\Data\data.grib'))

nr_years = len(grib_files) # number of years to process

# GLOBAL VARIABLES

# target spatial size (change as needed)
TARGET_Y = 80  # Take that as number of samples rather than according to coordinates
TARGET_X = 80
RF = 4  # Reduction Factor: take every RF-th pixel
TRAINING = True # Set to False to skip training and only do inference with pre-trained model
CONVERTING_DATA = True # Set to True to do the data extraction from ERA5 GRIB files to numpy arrays, else load pre-saved arrays
# Folder to save ML result images (use raw string for Windows path)

# Domain of interest
LON_COORDS, LAT_COORDS = [-5,15],[45,65]
TIME_RANGE = ["2021-01-01","2021-01-04"]  # Full year of 2024

RESULTS_DIR =os.path.join(BASE_DIR, "Results_ML/Try_Remote")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# ERA5 variables should be split into two groups
# ================================
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
        #print(f"Loading {f} for variables {variables} ...")
        ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs=backend_kwargs,
                             chunks={"time": 365})
        #print(f"{f} loaded with variables: {list(ds.data_vars.keys())}")
        datasets.append(ds)
    
    # Concatenate along time and sort to ensure correct temporal order
    combined = xr.concat(datasets, dim="time").sortby("time")
    print(f"Combined dataset for variables {variables}:")
    print(combined)
    print("\n")
    
    return combined

def coarsening_method(sample, rf):
    """Coarsen a (lat, lon, 2) sample by taking every rf-th pixel. It is equivalent to a downsampling"""
    # Need to add gaussian blur first -> Act as a low-pass filter to smooth things out ? 
    return sample[::rf, ::rf, :]  # Downsample spatial dimensions

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
    
    #plt.show()

def create_wind_animation(arrays_np_hr, arrays_np_lr, lat_min, lat_max, lon_min, lon_max,time_coords=None, n_frames=56, gif_path=os.path.join(RESULTS_DIR,"wind_animation_V2.gif")):
    """"Create animation of a sub-sample of a TensorFlow dataset containing wind data (u, v components)."""
    # Create GIF animation from the first 60 samples (or fewer if not enough)
    n_frames = min(56, arrays_np_hr.shape[0])
    if n_frames == 0:
        print("No frames available to animate.")
        return

    # Precompute global color limits for consistent frames, from the HR resolution Dataset
    u_all = arrays_np_hr[:n_frames, ..., 0].ravel()
    v_all = arrays_np_hr[:n_frames, ..., 1].ravel()
    speed_all = np.sqrt(u_all**2 + v_all**2)

    uv_vmin, uv_vmax = np.nanpercentile(np.concatenate([u_all, v_all]), [1, 99])
    speed_vmin, speed_vmax = np.nanpercentile(speed_all, [1, 99])

    # Use PlateCarree projection for all subplots so coastlines can be added
    fig, axes = plt.subplots(2, 3, figsize=(15, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # initial data for hr
    u_HR = arrays_np_hr[0, ..., 0]
    v_HR = arrays_np_hr[0, ..., 1]
    speed_HR = np.sqrt(u_HR**2 + v_HR**2)

    # Time setup for counter
    # Add time counter text (top-center of the figure)
    if time_coords is not None:
        time_text = fig.text(0.5, 0.05, '', ha='center', va='top', fontsize=15)
    else:
        time_text = None

    im0_HR = axes[0,0].imshow(u_HR, origin="lower", cmap="RdBu_r",
                        vmin=uv_vmin, vmax=uv_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[0,0].set_title("Zonal wind (U) at 100m")
    axes[0,0].coastlines()

    im1_HR = axes[0,1].imshow(v_HR, origin="lower", cmap="RdBu_r",
                        vmin=uv_vmin, vmax=uv_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[0,1].set_title("Meridional Wind (V) at 100m")
    axes[0,1].coastlines()

    im2_HR = axes[0,2].imshow(speed_HR, origin="lower", cmap="viridis",
                        vmin=speed_vmin, vmax=speed_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[0,2].set_title("Resulting Wind Speed")
    axes[0,2].coastlines()

    cbar0_HR = fig.colorbar(im0_HR, ax=axes[0,0], fraction=0.046, pad=0.04)
    cbar1_HR = fig.colorbar(im1_HR, ax=axes[0,1], fraction=0.046, pad=0.04)
    cbar2_HR = fig.colorbar(im2_HR, ax=axes[0,2], fraction=0.046, pad=0.04,label="[m/s]")

    # initial data for lr
    u_LR = arrays_np_lr[0, ..., 0]
    v_LR = arrays_np_lr[0, ..., 1]
    speed_LR = np.sqrt(u_LR**2 + v_LR**2)

    im0_LR = axes[1,0].imshow(u_LR, origin="lower", cmap="RdBu_r",
                        vmin=uv_vmin, vmax=uv_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[1,0].set_title("Coarsened U-component")
    axes[1,0].coastlines()

    im1_LR = axes[1,1].imshow(v_LR, origin="lower", cmap="RdBu_r",
                        vmin=uv_vmin, vmax=uv_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[1,1].set_title("Coarsened V-component")
    axes[1,1].coastlines()

    im2_LR = axes[1,2].imshow(speed_LR, origin="lower", cmap="viridis",
                        vmin=speed_vmin, vmax=speed_vmax,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        transform=ccrs.PlateCarree(), aspect="auto")
    axes[1,2].set_title("Coarsened speed")
    axes[1,2].coastlines()

    cbar0_LR = fig.colorbar(im0_LR, ax=axes[1,0], fraction=0.046, pad=0.04)   
    cbar1_LR = fig.colorbar(im1_LR, ax=axes[1,1], fraction=0.046, pad=0.04)
    cbar2_LR = fig.colorbar(im2_LR, ax=axes[1,2], fraction=0.046, pad=0.04,label="[m/s]")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(frame):
        # Upper part with the hr dataset
        u_HR = arrays_np_hr[frame, ..., 0]
        v_HR = arrays_np_hr[frame, ..., 1]
        s_HR = np.sqrt(u_HR**2 + v_HR**2)
        im0_HR.set_data(u_HR)
        im1_HR.set_data(v_HR)
        im2_HR.set_data(s_HR)

        # Lower part with the lr dataset
        u_LR = arrays_np_lr[frame, ..., 0]
        v_LR = arrays_np_lr[frame, ..., 1]
        s_LR = np.sqrt(u_LR**2 + v_LR**2)

        im0_LR.set_data(u_LR)
        im1_LR.set_data(v_LR)
        im2_LR.set_data(s_LR)

        # Update time counter if available
        if time_text is not None and time_coords is not None:
            current_time = time_coords[frame]
            # Format as string (adjust if needed, e.g., for datetime)
            if hasattr(current_time, 'strftime'):  # If it's a datetime
                time_str = current_time.strftime("%d/%m/%Y, %H:%M:%S")
            else:
                time_str = str(current_time)
            time_text.set_text(f'Timestamp: {time_str}')

        return (im0_HR, im1_HR, im2_HR, im0_LR, im1_LR, im2_LR)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=166, blit=True)

    writer = animation.PillowWriter(fps=6)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF animation with {n_frames} frames to: {gif_path}")

# ================================
# Computation of the quantities of interest: wind speed and wind direction
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

def domain_crop(df, lat_min, lat_max, lon_min, lon_max):

    print("Cropping to domain of interest...")
    print("dimensions before cropping:", df.shape)
    print(df[0,:,:])

    df_filtered = df.sel(
    latitude=slice(lat_max,lat_min),
    longitude=slice(lon_min,lon_max)
    )
    print("dimensions after cropping:", df_filtered.shape)
    # Plot the first time step of df and df_filtered side by side
 

    # To plot both domains (before and after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df.isel(time=0).plot(ax=axes[0])
    axes[0].set_title('Original Domain')
    df_filtered.isel(time=0).plot(ax=axes[1])
    axes[1].set_title('Cropped Domain')
    plt.tight_layout()
    plt.show()

    return df_filtered


def time_crop(df, start_time, end_time):
    """
    Crop the dataset to a specific time range.
    
    Parameters:
        df: xarray.DataArray or Dataset with 'time' dimension
        start_time: str or datetime, start of the time range
        end_time: str or datetime, end of the time range
    
    Returns:
        xarray.DataArray or Dataset cropped to the specified time range
    """
    print(f"Cropping to time range: {start_time} to {end_time}")
    df_cropped = df.sel(time=slice(start_time, end_time))
    print(f"Dimensions after time cropping: {df_cropped}")
    return df_cropped


# ================================
# Load surface and wind datasets
# ================================

# Define the SSIM and PSNR as metrics 
# Define and register custom PSNR metric
@tf.keras.utils.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    # PSNR: Peak Signal-to-Noise Ratio (higher is better)
    return tf.image.psnr(y_true, y_pred,max_val=15.0)  # Assuming inputs are normalized [0,1]

# Define and register custom SSIM metric
@tf.keras.utils.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    # SSIM: Structural Similarity Index (higher is better, 1.0 is perfect)
    return tf.image.ssim(y_true, y_pred,max_val=15.0)  # Assuming inputs are normalized [0,1]

# List the paths accessible to save lat/lon values
lat_lr_path = os.path.join(RESULTS_DIR,"lat_LR.npy")
lon_lr_path = os.path.join(RESULTS_DIR,"lon_LR.npy")

lat_hr_path = os.path.join(RESULTS_DIR,"lat_HR.npy")
lon_hr_path = os.path.join(RESULTS_DIR,"lon_HR.npy")

#Counter settings 


class SectionTimer:
    def __init__(self):
        self._starts = {}
        self.times = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        self.times[name] = time.perf_counter() - self._starts[name]


# ------------------ RUN START ------------------

run_start = datetime.now()
run_start_str = run_start.isoformat(timespec="seconds")

timer = SectionTimer()
total_start = time.perf_counter()


##################################
#
#          MAIN LOOP
#
##################################

if CONVERTING_DATA:
    # ---------------- PRE-PROCESSING ----------------
    timer.start("preprocessing")

    print("Starting data extraction from ERA5 GRIB files...")
# Load wind variables at 100m
    for f in grib_files:  # Loop over each file
        # Load wind variables at 100m
        print("Loading wind variables at 100m...")
        try:
            ds_wind = load_era5_variables([f], wind_vars)  # Adjust to load one file; assumes function expects a list
        except OSError as e:
            if "Invalid argument" in str(e):
                print(f"Skipping corrupted file {f}: {e}")
                continue
            else:
                raise

    print("Loaded wind variables at 100m...")
    print(psutil.virtual_memory()) 

    print("Converting to dataframes...")
    # Transform the dataset in x_Array as a numpy array
    ds_wind = time_crop(ds_wind, TIME_RANGE[0], TIME_RANGE[1])
    df_wind = ds_wind.to_dataframe(dim_order=None)
    print(df_wind.head())

    #df_wind.to_csv('DF_Wind.csv', index=True)

    # ...existing code...


    # Build (lat, lon, 2) numpy arrays per time step from the xarray dataset (safer than parsing df_wind)
    # Ensure ds_wind contains 'u100' and 'v100' with dims (time, latitude, longitude)
    u_da = ds_wind["u100"]
    v_da = ds_wind["v100"]
    n_frames = 56

    ds_wind["time"] = ds_wind["time"].dt.strftime('%Y-%m-%d %H:%M:%S')


    u_da = domain_crop(u_da, LAT_COORDS[0],LAT_COORDS[1], LON_COORDS[0],LON_COORDS[1])
    v_da = domain_crop(v_da, LAT_COORDS[0],LAT_COORDS[1], LON_COORDS[0],LON_COORDS[1])

    time_coords = ds_wind["time"].values[:n_frames]  # Slice to match n_frames

    #Change the format of the dataetime object into "%d/%m/%Y, %H:%M:%S"

    print("Computing lat/lon...")
    # ensure dataset is large enough
    ny = u_da.sizes.get("latitude", None)
    nx = u_da.sizes.get("longitude", None)
    if ny is None or nx is None:
        raise RuntimeError("Expected dims 'latitude' and 'longitude' in u_da/v_da")
    if ny < TARGET_Y or nx < TARGET_X:
        raise ValueError(f"The dataset too small: available (lat,lon)=({ny},{nx}), but requested ({TARGET_Y},{TARGET_X})")

    # keep the first TARGET_Y / TARGET_X points (removes the last lat/lon entries)
    #u_da = u_da.isel(latitude=slice(0, TARGET_Y), longitude=slice(0, TARGET_X))
    #v_da = v_da.isel(latitude=slice(0, TARGET_Y), longitude=slice(0, TARGET_X))
    #Filter by to only take the region of interest, e.g. Europe

    print("Building numpy arrays...")
    arrays = []
    for t in ds_wind["time"].values: # Do the same thing for each time step
        u_t = u_da.sel(time=t).values.astype(np.float32)   # shape (lat, lon)
        v_t = v_da.sel(time=t).values.astype(np.float32)   # shape (lat, lon)
        pair = np.stack([u_t, v_t], axis=-1)               # shape (lat, lon, 2), axis=-1 to stack according to the last axis
        arrays.append(pair)  # Build the list of samples surely but slowly



    # Stack into shape (time, lat, lon, 2)
    arrays_np_hr = np.stack(arrays, axis=0).astype(np.float32) # shape (time, lat, lon, 2), stack according to time, so the first dimension (axis=0) 
    # Plot one example of arrays_np_hhr to check

    print("Downsampling to create LR dataset...")
    arrays_np_lr = arrays_np_hr[:,::RF, ::RF, :]  # Downsampled version for LR dataset
    # need to check the equivalent to this 
    print("Final numpy array shape for HR:", arrays_np_hr.shape)  # Should be (time, lat, lon, 2), check the dimensions
    print("Final numpy array shape for LR:", arrays_np_lr.shape) 

    # Create the TensorFlow Dataset for the High Resolution data
    tf_ds_hr_unshuffled = tf.data.Dataset.from_tensor_slices(arrays_np_hr)
    print("High Resolution dataset created < ERA5.")

    # Create LR (Low Resolution) dataset by coarsening it using the Rescaling Factor (RF)
    tf_ds_lr_unshuffled = tf_ds_hr_unshuffled.map(lambda x: coarsening_method(x, RF))
    print("Low Resolution dataset created < ERA5Coarsed.")
    
    dataset = tf.data.Dataset.zip((tf_ds_lr_unshuffled, tf_ds_hr_unshuffled))

    buffer_size = 3000  # usually >= dataset size for full shuffle (sample size = 2920)
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=False,seed=16)

# If need them separately again
    tf_ds_lr = dataset.map(lambda x, y: x)
    tf_ds_hr = dataset.map(lambda x, y: y)

    # Plot the first five samples (u, v and speed) instead of printing shapes   # Then, will downscale the whole Tensorflow dataset instead.
    lat = ds_wind["latitude"].values
    lon = ds_wind["longitude"].values

    #Checking the region of interest 

    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat)) # Keep the latitude/longitude values to plot them.
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_small = ds_wind["latitude"].values[:TARGET_Y] # Keep the small lats/lons for ML postprocessing visualisation, since the image samples are cropped for proper sizing purposes.
    lon_small = ds_wind["longitude"].values[:TARGET_X]

    # Coarsened coords for LR
    lat_lr = lat_small[::RF]
    lon_lr = lon_small[::RF]

    # Remove this line if want to visualize the transformations as an animation
    #create_wind_animation(arrays_np_hr,arrays_np_lr, lat_min, lat_max, lon_min, lon_max,time_coords=time_coords)

    print("Saving the latitude and longitude values into .npy files")

    np.savetxt(lat_lr_path,lat_lr)
    np.savetxt(lon_lr_path,lon_lr)

    # Do the same thing for hr values
    np.savetxt(lat_hr_path,lat_small)
    np.savetxt(lon_hr_path,lon_small)

    timer.stop("preprocessing")



else:
    # ---------------- PRE-PROCESSING ----------------
    timer.start("preprocessing")

    # Import the lat/lon values from the saved files 
    lat_small = np.loadtxt(lat_hr_path)
    lon_small = np.loadtxt(lon_hr_path)

    lat_lr = np.loadtxt(lat_lr_path)
    lon_lr = np.loadtxt(lon_lr_path)

    timer.stop("preprocessing")



    # Remove this line if want to visualize the transformations as an animation
    #create_wind_animation(arrays_np_hr,arrays_np_lr, lat_min, lat_max, lon_min, lon_max,time_coords=time_coords)


    # For data management purposes, save the two big arrays into files to not redo the painful extracting process again

    # Function to save large NumPy array with coords in chunks as CSV
def save_array_in_chunks(arr, lat_coords_1d, lon_coords_1d, filename_prefix, chunk_size=1000000):
    time_steps, n_lat, n_lon, _ = arr.shape
    # Create coord arrays (broadcasted)
    time_coords = np.repeat(np.arange(time_steps), n_lat * n_lon)
    lat_coords = np.tile(np.repeat(lat_coords_1d, n_lon), time_steps)
    lon_coords = np.tile(np.tile(lon_coords_1d, n_lat), time_steps)
    u_vals = arr[..., 0].ravel()
    v_vals = arr[..., 1].ravel()
    # Stack into 2D array: (samples, 5)
    data = np.column_stack((time_coords, lat_coords, lon_coords, u_vals, v_vals))
    
    if chunk_size is None or data.shape[0] <= chunk_size:
        # Save as single CSV
        np.savetxt(f'{filename_prefix}.csv', data, delimiter=',', header='time,lat,lon,u100,v100', comments='')
        print(f"Saved single CSV: {filename_prefix}.csv")
    else:
        # Save in chunks
        n_chunks = (data.shape[0] // chunk_size) + 1
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, data.shape[0])
            chunk = data[start:end]
            np.savetxt(f'{filename_prefix}_chunk_{i}.csv', chunk, delimiter=',', header='time,lat,lon,u100,v100', comments='')
        print(f"Saved {n_chunks} chunks for {filename_prefix}")


# Save arrays as CSV in chunks of 1 million rows, later can be reloaded in pandas dataframes easily
# TO DO & Figure out how to do it


#%%
# ================================
# ML Algorithm part
# ================================

# -------------------------------------------------------------------
# 1. SCALING FUNCTION -- To check the scaling of the datasets
# -------------------------------------------------------------------
def scaling(tf_ds_lr, tf_ds_hr):
    """
    Scales LR and HR datasets using sklearn MinMaxScaler.
    Returns scaled numpy arrays and the fitted scalers.
    """

    lr = np.array(list(tf_ds_lr))   # (112, 41, 41, 2)
    hr = np.array(list(tf_ds_hr))   # (112, 161, 161, 2)

    lr_flat = lr.reshape(-1, lr.shape[-1]) 
    hr_flat = hr.reshape(-1, hr.shape[-1])
    print("LR shape after flattening, before scaling: ", lr_flat.shape)
    print("HR shape after flattening, before scaling:", hr_flat.shape)

    # Use the min-max normalisation, might be less robust when dealing with outliers, but could also try StandardScaler later
    scaler_lr = MinMaxScaler()
    scaler_hr = MinMaxScaler()

    lr_scaled = scaler_lr.fit_transform(lr_flat).reshape(lr.shape)
    hr_scaled = scaler_hr.fit_transform(hr_flat).reshape(hr.shape)

    print("Arrays scaled with Min-Max normalisation.")
    return lr_scaled, hr_scaled, scaler_lr, scaler_hr

# -------------------------------------------------------------------
# 2. SRCNN MODEL FOR SUPER-RESOLUTION 
#    (Deep CNN, Architecture taken from https://arxiv.org/pdf/1501.00092, with 2-channels (U/V components), and took [n1=64, n2=32] [f1=9, f2=1, f3=5].)
# -------------------------------------------------------------------
def build_SRCNN():
    """
    SRCNN architecture for 2-channel images.  # What would be the difference when using 3D arrays with a single channel and this ? Are there specific properties /methods associated to it?
    Input resolution: 40×40×2
    Output resolution: 160×160×2 (done after initial upsampling) 
    """

    model = keras.Sequential([
        layers.Input(shape=(40, 40, 2)),

        # First: Upsample LR → HR resolution
        layers.UpSampling2D(size=(4, 4), interpolation='bilinear'), # Actually get the original picture right away, so a bit cheating? REplaced with bilinear interpolation instead.

        # SRCNN layers
        # First convolutional layer, want 64 filters, 9x9 kernel, ReLU activation
        layers.Conv2D(64, (9, 9), activation='relu', padding="same"),
        # Second convolutional layer, want 32 filters instead, 3x3 kernel (or 1x1 for base case and fastest), ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        # Last layer, get back to the right dimensions, or extract the overall result.
        layers.Conv2D(2, (5, 5), activation='linear', padding="same")
        # Get a lower dimension because no padding assumed in original SRCNN paper, since the region on interest is broader, is it a problem ?
        # We could recrop to get the original size reduced and apply the metrics ?
        # Use paddings for now for units consistency (set padding="same" instead of "valid"), but it will results in border effects (visible) that have to be accounted for + discussed later.
    ])


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss='mse',
        metrics=[
        'mse',  # Keep MSE if desired
        psnr_metric,  # PSNR metric (higher is better)
        ssim_metric   # SSIM metric (higher is better, 1.0 is perfect)
    ]
    )

    model.build(input_shape=(None, 40, 40, 2))

    # print layer shapes + params
    model.summary()

    # optional: print each layer's input/output shapes explicitly
    for layer in model.layers:
        print(f"{layer.name}: input_shape={getattr(layer, 'input_shape', None)}, output_shape={getattr(layer, 'output_shape', None)}")

    return model
# -------------------------------------------------------------------
# 3. TRAINING + PLOTTING FUNCTION
# -------------------------------------------------------------------
def train_SRCNN(tf_ds_lr, tf_ds_hr):

    # Scale datasets
    lr_scaled, hr_scaled, scaler_LR, scaler_HR = scaling(tf_ds_lr, tf_ds_hr)

    # Train / Test / Validation split
    N = lr_scaled.shape[0]
    n_train = int(0.70 * N)
    n_test = int(0.15 * N) # 15% for testing and 15% for validation

    #Split into the different datasets, with their associated inputs and labels
    X_train = lr_scaled[:n_train]
    y_train = hr_scaled[:n_train]

    X_test = lr_scaled[n_train:n_train+n_test]
    y_test = hr_scaled[n_train:n_train+n_test]

    X_val = lr_scaled[n_train+n_test:]
    y_val = hr_scaled[n_train+n_test:]

    print("Datasets split into Train / Test / Validation.")
    # Create SRCNN
    model = build_SRCNN()

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model, history, scaler_LR, scaler_HR, X_test, y_test

def test_model(X_tst_scaled, y_tst_scaled, model, scaler_X, scaler_y,history):
    """
    Test the trained SRCNN model on the test dataset and print evaluation metrics.
    """
    
    # Test metrics
    test_metrics = model.evaluate(X_tst_scaled, y_tst_scaled, verbose=0)
    test_loss, test_mse, test_psnr, test_ssim = test_metrics
    # To check the performances, the metrics are already implemented in Tensorflow

    print("\n--- TEST METRICS ---")
    print(f"Test Loss (MSE): {test_loss:.3f}")
    print(f"Test PSNR:       {test_psnr:.3f}")
    print(f"Test SSIM:       {test_ssim:.3f}\n")

    # Compute the number of epochs to follow a general trend
    epochs_trained = len(history.history.get("loss", []))

    # Now let's save them into a running .csv file to get the evolution of the metrics with different architectures / hyperparameters
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
    new_row = [datetime.now().isoformat(), np.round(test_loss,6), np.round(test_mse,6), np.round(test_psnr,6), np.round(test_ssim,6),int(epochs_trained)]

    # create file with header if it doesn't exist, then append
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_time", "test_loss", "test_mse", "test_psnr", "test_ssim", "epochs"])

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

    print(f"Saved test metrics to: {metrics_path}")
    epochs_plot = np.linspace(0,epochs_trained,1)
    # ------------------------------------
    # Plot Loss History
    # ------------------------------------
    # existing plotting code for losses (keeps it)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.hlines(test_loss, 0, len(history.history["loss"]),
               colors="black", linestyles="dashed",
               label=f"Test Loss = {test_loss:.5f}")

    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.ylim(0,0.004)
    plt.title("Training / Validation / Test Loss")
    plt.legend()
    plt.grid()

    # Save loss history figure
    loss_path = os.path.join(RESULTS_DIR, "loss_history.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    print(f"Saved loss history figure: {loss_path}")

    #plt.show()


    # prediction from the model on the test set
    y_predicted_test_scaled = model.predict(X_tst_scaled, verbose=0)

    # Inverse scale predictions and ground truth for evaluation
    y_predicted_flat = y_predicted_test_scaled.reshape(-1, y_predicted_test_scaled.shape[-1])
    y_predicted_test_unscaled = scaler_y.inverse_transform(y_predicted_flat).reshape(y_predicted_test_scaled.shape)

    # Inverse scale predictions and ground truth
    X_test_flat = X_tst_scaled.reshape(-1, X_tst_scaled.shape[-1])
    y_test_flat = y_tst_scaled.reshape(-1, y_tst_scaled.shape[-1])

    X_test_unscaled = scaler_X.inverse_transform(X_test_flat).reshape(X_tst_scaled.shape)
    y_test_unscaled = scaler_y.inverse_transform(y_test_flat).reshape(y_tst_scaled.shape)

    # Compute evaluation metrics
    mse = np.mean((y_predicted_test_unscaled - y_test_unscaled) ** 2)
    psnr = tf.image.psnr(y_test_unscaled, y_predicted_test_unscaled, max_val=np.max(y_test_unscaled)).numpy().mean()
    ssim = tf.image.ssim(y_test_unscaled, y_predicted_test_unscaled, max_val=np.max(y_test_unscaled)).numpy().mean()

    print("\n--- EVALUATION ON TEST DATA ---")
    print(f"MSE:  {mse:.6f}")
    print(f"PSNR: {psnr:.6f}")
    print(f"SSIM: {ssim:.6f}\n")

    # Return the scaled samples to plot them
    return y_predicted_test_unscaled, y_test_unscaled, X_test_unscaled

def plot_comparison(lr_unscaled, sr_unscaled, hr_unscaled, lat=None, lon=None, index=0):
    """
    Inverse transforms LR, SR, and HR arrays using fitted scalers to get them back to original magnitude.
    Plots LR, SR, and HR side-by-side for visual comparison.

    Parameters
    ----------
    lr_scaled : np.array
        Scaled low-resolution dataset (N, 41, 41, channels)
    sr_scaled : np.array
        Scaled super-resolution predictions (N, 161, 161, channels)
    hr_scaled : np.array
        Scaled high-resolution dataset (N, 161, 161, channels)
    scaler_lr : MinMaxScaler
        Scaler fitted on LR dataset
    scaler_hr : MinMaxScaler
        Scaler fitted on HR dataset
    index : int
        Index of the sample to visualize
    """

    # ---------------------------------------------------------
    # 1. Pull one sample
    # ---------------------------------------------------------
    lr_img  = lr_unscaled[index]
    sr_img  = sr_unscaled[index]
    hr_img  = hr_unscaled[index]
    print(f"Shapes of selected images (scaled): LR {lr_img.shape}, SR {sr_img.shape}, HR {hr_img.shape}")
    # ---------------------------------------------------------
    # 3. Prepare LR for visualization: upscale for plotting
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 4. Plot LR (upscaled), SR, and HR
    # ---------------------------------------------------------
    # plot LR (upscaled), SR, and HR with individual colorbars (consistent limits)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5),
                           subplot_kw={'projection': ccrs.PlateCarree()} if lat is not None and lon is not None else None)

    # choose channel to display (0 = u component, 1 = v component)
    ch = 0

    # compute common color limits across the three images for consistent colorbars
    all_vals = np.concatenate([
        lr_img[..., ch].ravel(),
        sr_img[..., ch].ravel(),
        hr_img[..., ch].ravel()
    ])
    vmin, vmax = np.nanpercentile(all_vals, [1, 99])

    if lat is None or lon is None:
    # fallback to previous imshow behavior (no coastlines)
        im0 = ax[0].imshow(lr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[0].set_title("Low-Res. input (Coarsened ERA5)", fontsize=16)
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="U-component [m/s]")

        im1 = ax[1].imshow(sr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[1].set_title("Downscaled prediction (SRCNN Output)", fontsize=16)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="U-component [m/s]")

        im2 = ax[2].imshow(hr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[2].set_title("Ground Truth (ERA5)", fontsize=16)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="U-component [m/s]")
    else:
    # make 2D lon/lat grids matching the HR arrays
        
        # prepare HR and LR coordinate grids
        lon_hr = np.asarray(lon)   # full HR 1D lon (length TARGET_X)
        lat_hr = np.asarray(lat)   # full HR 1D lat (length TARGET_Y)

        # Get also the specific values for the lr dataset which is smaller size;
        # LR coords: subsample by RF
        lon_lr = lon_hr[::RF]
        lat_lr = lat_hr[::RF]

        #Create the meshgrid for both datasets
        lon2d_hr, lat2d_hr = np.meshgrid(lon_hr, lat_hr)
        lon2d_lr, lat2d_lr = np.meshgrid(lon_lr, lat_lr) # HR grid for SR and HR images


        im0 = ax[0].pcolormesh(lon2d_lr, lat2d_lr, lr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax[0].set_title("Low-Res. input (Coarsened ERA5)", fontsize=16)
        ax[0].coastlines(resolution='50m')
        ax[0].add_feature(cfeature.BORDERS, linestyle=':')
        ax[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="U-component [m/s]")

        im1 = ax[1].pcolormesh(lon2d_hr, lat2d_hr, sr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax[1].set_title("Downscaled prediction (SRCNN Output)", fontsize=16)
        ax[1].coastlines(resolution='50m')
        ax[1].add_feature(cfeature.BORDERS, linestyle=':')
        ax[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="U-component [m/s]")

        im2 = ax[2].pcolormesh(lon2d_hr, lat2d_hr, hr_img[..., ch], cmap="RdBu_r", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax[2].set_title("Ground Truth (ERA5)", fontsize=16)
        ax[2].coastlines(resolution='50m')
        ax[2].add_feature(cfeature.BORDERS, linestyle=':')
        ax[2].add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="U-component [m/s]")


    # In any case, keep the same title and the same layout
    plt.suptitle(f"Input / Prediction / Ground truth Comparison", fontsize=17)
    plt.tight_layout()

    # Extra lines to save automatically the figures as .png files
    # Save comparison figure
    cmp_path = os.path.join(RESULTS_DIR, f"comparison_index_{index:03d}_Filtered_Domain.png")
    plt.savefig(cmp_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure: {cmp_path}")

    #plt.show()


# -------------------------------------------------------------------
# Get model, history and scalers after training
# -------------------------------------------------------------------

if TRAINING:
    
    # ------------------ TRAINING --------------------  
    timer.start("training")

    model, history1,sclr_lr,sclr_hr,x_test,Y_test = train_SRCNN(tf_ds_lr, tf_ds_hr)

    # Save all the output parameters after training (model, history, & test datasets with associated scaling)
    model_path = os.path.join(RESULTS_DIR, "srcnn_model.keras")
    model.save(model_path)  # The file needs to end with the .keras extension
    print(f"Saved trained SRCNN model to: {model_path}")
    # Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.npy")
    np.save(history_path, history1.history)
    print(f"Saved training history to: {history_path}")

    # Save scalers
    scaler_lr_path = os.path.join(RESULTS_DIR, "scaler_lr_or_X.pkl")
    with open(scaler_lr_path, "wb") as f:
        pickle.dump(sclr_lr, f)
    print(f"Saved LR scaler to: {scaler_lr_path}")
    scaler_hr_path = os.path.join(RESULTS_DIR, "scaler_hr_or_Y.pkl")
    with open(scaler_hr_path, "wb") as f:
        pickle.dump(sclr_hr, f)
    print(f"Saved HR scaler to: {scaler_hr_path}")

    # Save the two test datasets for postprocessing visualisation later
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), x_test)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), Y_test)
    print(f"Saved test datasets to: {RESULTS_DIR}")
    
    timer.stop("training")

else: # Load the model and associated parameters as they are already trained
    # Load model
    # ------------------ TRAINING --------------------  
    timer.start("training")


    model_path = os.path.join(RESULTS_DIR, "srcnn_model.keras")
    model = keras.models.load_model(model_path)
    # Manually compile the model with the original loss/optimizer/metrics (adjust as needed)
    model.compile(
       optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss='mse',
        metrics=[
        'mse',  # Keep MSE if desired
        psnr_metric,  # PSNR metric (higher is better)
        ssim_metric   # SSIM metric (higher is better, 1.0 is perfect)
    ])
    print(f"Loaded trained SRCNN model from: {model_path}")
    # Load training history
    history_path = os.path.join(RESULTS_DIR, "training_history.npy")
    history_dict = np.load(history_path, allow_pickle=True).item()
    history1 = keras.callbacks.History()
    history1.history = history_dict
    print(f"Loaded training history from: {history_path}")
    # Load scalers
    scaler_lr_path = os.path.join(RESULTS_DIR, "scaler_lr_or_X.pkl")
    with open(scaler_lr_path, "rb") as f:
        sclr_lr = pickle.load(f)
    print(f"Loaded LR scaler from: {scaler_lr_path}")
    scaler_hr_path = os.path.join(RESULTS_DIR, "scaler_hr_or_Y.pkl")
    with open(scaler_hr_path, "rb") as f:
        sclr_hr = pickle.load(f)
    print(f"Loaded HR scaler from: {scaler_hr_path}")
    # Load test datasets
    x_test = np.load(os.path.join(RESULTS_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
    print(f"Loaded test datasets from: {RESULTS_DIR}")

    timer.stop("training")


#Load the model (if needed)
#model = keras.models.load_model(model_path, compile=False)
# Make predictions on the scaled dataset


# ------------------ SAMPLING --------------------
timer.start("sampling")


# Test the model and plot loss history
y_predicted_test_unscaled, y_test_unscaled, X_test_unscaled = test_model(x_test, Y_test, model, sclr_lr, sclr_hr, history1)

timer.stop("sampling")

# Compare sample #0 (or any index)
plot_comparison(X_test_unscaled, y_predicted_test_unscaled,y_test_unscaled, lat=lat_small,lon=lon_small, index=9)







# ------------------ TOTAL -----------------------
total_time = time.perf_counter() - total_start

#  CSV LOGGING of different times to gather 


csv_file =os.path.join(RESULTS_DIR,"computation_time_runs.csv")
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "run_start_time",
            "preprocessing_seconds",
            "training_seconds",
            "sampling_seconds",
            "total_seconds"
        ])

    writer.writerow([
        run_start_str,
        f"{timer.times['preprocessing']:.4f}",
        f"{timer.times['training']:.4f}",
        f"{timer.times['sampling']:.4f}",
        f"{total_time:.4f}"
    ])
