#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/01/26

@author: Marion Falkenauer
Data matching between ERA5 and CERRA to obtain a visual idea of how to perform the first downscaling step
"""
# Package import
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
import os
import csv
from datetime import datetime
import pickle
import pandas as pd
import time # Module for counter
import psutil


# Plot settings (need to fix the font issue for the font family)
plt.rcParams.update({
    "font.size": 25,           # default text size
    "axes.labelsize": 20,      # x and y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    #'font.family': 'Caladea',
    "axes.titlesize": 20
})



# ================================
# Configuration
# ================================
# Base directory containing everything
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to a file in the same directory
ERA5_files = os.path.join(BASE_DIR, "Data/ERA5/era5_single_2024_3h.grib")
CERRA_files = os.path.join(BASE_DIR, "Data/CERRA/CERRA_100m_3h-Jan-2024.nc")
# Working directory containing ERA5 GRIB files
# Line for the yearly prediction
grib_files = sorted(glob.glob(ERA5_files))
grib_files_V2 = sorted(glob.glob(CERRA_files))

nr_years = len(grib_files) # number of years to process

# GLOBAL VARIABLES

# target spatial size (change as needed)
TARGET_Y = 80  # Take that as number of samples rather than according to coordinates
TARGET_X = 80
RF = 4  # Reduction Factor: take every RF-th pixel
TRAINING = False # Set to False to skip training and only do inference with pre-trained model
CONVERTING_DATA = True # Set to True to do the data extraction from ERA5 GRIB files to numpy arrays, else load pre-saved arrays
# Folder to save ML result images (use raw string for Windows path)

# Domain of interest
LON_COORDS, LAT_COORDS = [-5,15],[45,65]
TIME_RANGE = ["2024-01-01","2024-01-02"]  # First 2 days of 2024

RESULTS_DIR =os.path.join(BASE_DIR, "Results_ML/Try_CERRA")

os.makedirs(RESULTS_DIR, exist_ok=True)

wind_vars = ["100u", "100v"]


def load_era5_variables(files, variables, level_type=None,file_type="cfgrib"):
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
        if file_type=="cfgrib":
            ds = xr.open_dataset(f, engine=file_type, backend_kwargs=backend_kwargs,
                                chunks={"time": 365})
        else:
            print(str(f))
            ds = xr.open_dataset(CERRA_files,engine=file_type)     
        #print(f"{f} loaded with variables: {list(ds.data_vars.keys())}")
        datasets.append(ds)
    
    # Concatenate along time and sort to ensure correct temporal order
    combined = xr.concat(datasets, dim="time").sortby("time")
    print(f"Combined dataset for variables {variables}:")
    print(combined)
    print("\n")
    
    return combined


def domain_crop(df, lat_min, lat_max, lon_min, lon_max):
    """
    Allow to crop the domain of interest for different datasets.
    
    :param df:  xarray.DataArray, the data before numpy/panda transformation
    :param lat_min: southern bound of latitude (min;0)
    :param lat_max: northern bound of latitude (max,90 at poles)
    :param lon_min: western bound for longitude (min,-5)
    :param lon_max: eastern bound for longitude (max,180)

    output:  xarray.DataArray  filtered spatially 
    """

    print("Cropping to domain of interest...")

    df_filtered = df.sel(
    latitude=slice(lat_max,lat_min),
    longitude=slice(lon_min,lon_max)
    )
    # Plot the first time step of df and df_filtered side by side
    """
    # To plot both domains (before and after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df.isel(time=1).plot("u100",ax=axes[0])
    axes[0].set_title('Original Domain')
    df_filtered.isel(time=1).plot("u100",ax=axes[1])
    axes[1].set_title('Cropped Domain')
    plt.tight_layout()
    plt.show()
    
    """
    print(df)
    print(df_filtered)
    return df_filtered


def time_crop(df, start_time, end_time):
    """
    Crop the dataset to a specific time range.
    
    Parameters:
        df: xarray.DataArray with 'time' dimension
        start_time: str or datetime, start of the time range
        end_time: str or datetime, end of the time range
    
    Returns:
        xarray.DataArray cropped to the specified time range
    """
    print(f"Cropping to time range: {start_time} to {end_time}")
    df_cropped = df.sel(time=slice(start_time, end_time))
    print(f"Dimensions after time cropping: {df_cropped}")
    return df_cropped

def coarsening_method(sample, rf):
    """Coarsen a (lat, lon, 2) sample by taking every rf-th pixel. It is equivalent to a downsampling"""
    # Need to add gaussian blur first -> Act as a low-pass filter to smooth things out ? 
    return sample[::rf, ::rf, :]  # Downsample spatial dimensions

def wind_speed_to_uv(df):
    """
    Converts 'wind_speed' and 'wind_direction' columns in df to 'u100' and 'v100'.
    Overwrites the original columns and returns the modified DataFrame.
    Assume the  wind_direction in degrees for now
    """
    # Convert direction to radians
    wd_rad = np.deg2rad(df['wind_direction'])
    # Need to check the direction (either put a minus or not in front of "df['wind_speed']")
    df['u100'] = df['wind_speed'] * np.sin(wd_rad)
    df['v100'] = df['wind_speed'] * np.cos(wd_rad)
    df = df.drop(['wind_speed', 'wind_direction'], axis=1)
    return df

# List the paths accessible to save lat/lon values
lat_ERA5_path = os.path.join(RESULTS_DIR,"lat_LR.npy")
lon_ERA5_path = os.path.join(RESULTS_DIR,"lon_LR.npy")

lat_CERRA_path = os.path.join(RESULTS_DIR,"lat_HR.npy")
lon_CERRA_path = os.path.join(RESULTS_DIR,"lon_HR.npy")

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
            print(CERRA_files)
            ds_wind_ERA5 = load_era5_variables([f], wind_vars,file_type="cfgrib")  # Adjust to load one file; assumes function expects a list
            ds_wind_CERRA = load_era5_variables(CERRA_files, wind_vars,file_type='netcdf4')
            print('Variable loaded')
        except OSError as e:
            if "Invalid argument" in str(e):
                print(f"Skipping corrupted file {f}: {e}")
                continue
            else:
                raise

    print("Loaded wind variables at 100m")

    print("Converting to dataframes...")
    # Transform the dataset in x_Array as a numpy array
    # Crop first according to the region and time of interest for computational savings
    ds_wind_ERA5 = time_crop(ds_wind_ERA5, TIME_RANGE[0], TIME_RANGE[1])
    ds_wind_CERRA = time_crop(ds_wind_CERRA, TIME_RANGE[0], TIME_RANGE[1])

    ds_wind_ERA5 = domain_crop(ds_wind_ERA5, LAT_COORDS[0],LAT_COORDS[1], LON_COORDS[0],LON_COORDS[1])
    ds_wind_CERRA = domain_crop(ds_wind_CERRA, LAT_COORDS[0],LAT_COORDS[1], LON_COORDS[0],LON_COORDS[1]) 

    df_wind_ERA5 = ds_wind_ERA5.to_dataframe(dim_order=None)
    df_wind_CERRA = ds_wind_CERRA.to_dataframe(dim_order=None)
    print(df_wind_ERA5.head())
    print(df_wind_CERRA.head())





    # Build (lat, lon, 2) numpy arrays per time step from the xarray dataset (safer than parsing df_wind)
    # Ensure ds_wind contains 'u100' and 'v100' with dims (time, latitude, longitude)
    u_da_ERA5 = ds_wind_ERA5["u100"]
    v_da_ERA5 = ds_wind_ERA5["v100"]

    u_da_CERRA = ds_wind_CERRA["u100"]
    v_da_CERRA= ds_wind_CERRA["v100"]
    n_frames = 56

    ds_wind_ERA5["time"] = ds_wind_ERA5["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
    ds_wind_CERRA["time"] = ds_wind_CERRA["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Drop the other name for time
    
    """"
    ds_wind["time"] = ds_wind["time"].dt.strftime('%Y-%m-%d %H:%M:%S')

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


    print("Only dimension check;")
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
    #tf_ds_lr = dataset.map(lambda x, y: x)
    #tf_ds_hr = dataset.map(lambda x, y: y)


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

    print("Saving the latitude and longitude values into .npy files")

    np.savetxt(lat_ERA5_path,lat_lr)
    np.savetxt(lon_ERA5_path,lon_lr)

    # Do the same thing for hr values
    np.savetxt(lat_CERRA_path,lat_small)
    np.savetxt(lon_CERRA_path,lon_small)

    """

    timer.stop("preprocessing")
    print('pre-processing finished')


