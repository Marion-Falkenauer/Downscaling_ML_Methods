"""Statistics of the ERA5 (.grib) and CERRA (.nc) wind components, written to a CSV file.

Each file is cropped to the domain and time range of interest and reduced to its 3-hourly samples
(same preprocessing as the training pipeline). The u and v components are min-max scaled with fixed
global bounds, then n, min, max, mean and variance of each component are stored per file in
CSV_LOG_PATH.

Configuration (module constants):
    ERA5_INPUT_DIR, CERRA_INPUT_DIR: folders of the input files.
    CSV_LOG_PATH: output CSV.
    LON_COORDS, LAT_COORDS, LON_COORDS_CERRA, LAT_COORDS_CERRA, TIME_RANGE: domain and period.
    ADD_COMBINED_STATS: True appends the statistics pooled over all years (combine_stats).
    
Author: Marion Falkenauer (with the help of Claude)
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

from Pre_Processing_Functions import domain_crop, time_crop, wind_speed_to_uv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

ERA5_INPUT_DIR = os.path.join(DATA_DIR, "ERA5")
CERRA_INPUT_DIR = os.path.join(DATA_DIR, "CERRA")


CSV_LOG_PATH = os.path.join(DATA_DIR, "statistics_2020-2024_min-max_normalised_already.csv")

LON_COORDS, LAT_COORDS = [-5,14.75],[44,59.75]
LON_COORDS_CERRA, LAT_COORDS_CERRA = [-5,14.95],[44,59.95]
TIME_RANGE = ["2020-01-01","2024-12-31"] # 4 years case

wind_vars = ["100u", "100v"]

ADD_COMBINED_STATS = False


def load_era5_variables(files, variables, level_type=None, file_type="cfgrib"):
    """Load GRIB or NetCDF files and concatenate them along time.

    Args:
        files: path or list of paths.
        variables: GRIB short names to read (cfgrib only).
        level_type: GRIB typeOfLevel filter (cfgrib only).
        file_type: "cfgrib", "netcdf4" or "h5netcdf".
    Returns:
        xarray.Dataset sorted by time.
    """
    if isinstance(files, str):
        files = [files]
    filter_keys = {"shortName": variables}
    if level_type is not None:
        filter_keys["typeOfLevel"] = level_type

    backend_kwargs = {"filter_by_keys": filter_keys}

    datasets = []

    for f in files:
        print(f"Loading {f} ...")

        if file_type == "cfgrib":
            ds = xr.open_dataset(
                f,
                engine="cfgrib",
                backend_kwargs=backend_kwargs
            )
        elif file_type in ["netcdf4", "h5netcdf"]:
            ds = xr.open_dataset(f, engine=file_type)
        else:
            raise ValueError("Unsupported file_type")

        datasets.append(ds)

    combined = xr.concat(datasets, dim="time").sortby("time")
    return combined


def extract_year(filename):
    """Return the first four-digit number found in a file name, or None."""
    import re
    match = re.search(r"\d{4}", filename)
    return int(match.group()) if match else None


def compute_stats_ERA5(u, v):
    """Statistics of ERA5 u and v after min-max scaling with the fixed ERA5 bounds.

    Args:
        u, v: arrays of the u and v components in m/s.
    Returns:
        Dictionary with n and the min, max, mean and variance of each scaled component.
    """
    minimum_computed_u = -30.670517
    maximum_computed_u = 37.827896
    u_prime = (u - minimum_computed_u)/(maximum_computed_u - minimum_computed_u)

    minimum_computed_v = -32.545272
    maximum_computed_v = 31.206726
    v_prime = (v - minimum_computed_v)/(maximum_computed_v - minimum_computed_v)

    n = u.size
    return {
        "n": int(n),
        "min_u": float(np.min(u_prime)),
        "max_u": float(np.max(u_prime )),
        "mean_u": float(np.mean(u_prime)),
        "variance_u": float(np.var(u_prime)),
        "min_v": float(np.min(v_prime)),
        "max_v": float(np.max(v_prime)),
        "mean_v": float(np.mean(v_prime)),
        "variance_v": float(np.var(v_prime)),
    }


def compute_stats_CERRA(u, v):
    """Statistics of CERRA u and v after min-max scaling with the fixed CERRA bounds.

    Args:
        u, v: arrays of the u and v components in m/s.
    Returns:
        Dictionary with n and the min, max, mean and variance of each scaled component.
    """
    minimum_computed_u = -35.775658
    maximum_computed_u = 40.71696
    u_prime = (u - minimum_computed_u)/(maximum_computed_u - minimum_computed_u)

    minimum_computed_v = -34.592712
    maximum_computed_v = 35.852985
    v_prime = (v - minimum_computed_v)/(maximum_computed_v - minimum_computed_v)

    n = u.size
    return {
        "n": int(n),
        "min_u": float(np.min(u_prime)),
        "max_u": float(np.max(u_prime )),
        "mean_u": float(np.mean(u_prime)),
        "variance_u": float(np.var(u_prime)),
        "min_v": float(np.min(v_prime)),
        "max_v": float(np.max(v_prime)),
        "mean_v": float(np.mean(v_prime)),
        "variance_v": float(np.var(v_prime)),
    }


def combine_stats(df, dataset_name):
    """Pool the per-file statistics of one dataset.

    Args:
        df: DataFrame with the per-file statistics.
        dataset_name: "ERA5" or "CERRA".
    Returns:
        Dictionary with the global min and max and the sample-weighted mean and variance of each component.
    """

    df_d = df[df["dataset"] == dataset_name]

    n = df_d["n"].values

    means_u = df_d["mean_u"].values
    vars_u  = df_d["variance_u"].values

    print(means_u)
    print(vars_u)
    combined_mean_u = np.sum(n * means_u) / np.sum(n)

    combined_var_u = np.sum(
        n * (vars_u + (means_u - combined_mean_u)**2)
    ) / np.sum(n)

    print(means_u)
    print(vars_u)

    means_v = df_d["mean_v"].values
    vars_v  = df_d["variance_v"].values

    combined_mean_v = np.sum(n * means_v) / np.sum(n)

    combined_var_v = np.sum(
        n * (vars_v + (means_v - combined_mean_v)**2)
    ) / np.sum(n)

    return {
        "dataset": dataset_name,
        "year": "2020-2024",

        "min_u": df_d["min_u"].min(),
        "max_u": df_d["max_u"].max(),
        "mean_u": combined_mean_u,
        "variance_u": combined_var_u,

        "min_v": df_d["min_v"].min(),
        "max_v": df_d["max_v"].max(),
        "mean_v": combined_mean_v,
        "variance_v": combined_var_v,
    }


def process_era5_file(file_path):
    """Read and preprocess one ERA5 file.

    Args:
        file_path: GRIB file.
    Returns:
        (data, u, v): array (T, H, W, 2) and the u and v arrays (T, H, W) in m/s.
    """

    print(f"\nProcessing ERA5: {file_path}")

    ds = load_era5_variables(file_path, wind_vars, file_type="cfgrib")

    # Crop to the time range and domain of interest
    ds = time_crop(ds, TIME_RANGE[0], TIME_RANGE[1], time_var="time")
    print(ds)
    ds = domain_crop(ds, LAT_COORDS[0], LAT_COORDS[1],
                     LON_COORDS[0], LON_COORDS[1], variable_name="long")

    # Keep the 3-hourly samples
    allowed_hours = [0,3,6,9,12,15,18,21]
    mask = ds["time"].dt.hour.isin(allowed_hours)
    ds = ds.isel(time=mask)

    # Extract u and v
    u = ds["u100"].values.astype(np.float32)
    v = ds["v100"].values.astype(np.float32)

    # Stack to (time, lat, lon, 2)
    data = np.stack([u, v], axis=-1)
    print("Processed ERA5 file looks like :")
    print(ds)
    return data, u, v


def process_cerra_file(file_path):
    """Read and preprocess one CERRA file.

    Args:
        file_path: NetCDF file.
    Returns:
        (data, u, v): array (T, H, W, 2) and the u and v arrays (T, H, W) in m/s.
    """

    print(f"\nProcessing CERRA: {file_path}")

    ds = load_era5_variables(file_path, wind_vars, file_type="h5netcdf")

    # Crop to the time range and domain of interest
    ds = time_crop(ds, TIME_RANGE[0], TIME_RANGE[1], time_var="time")
    print(ds)
    ds = domain_crop(ds,
                     LAT_COORDS_CERRA[0], LAT_COORDS_CERRA[1],
                     LON_COORDS_CERRA[0], LON_COORDS_CERRA[1],
                     variable_name="short")

    ds = wind_speed_to_uv(ds)

    # Reverse the latitude axis
    ds = ds.sel(lat=slice(None, None, -1))

    # Extract u and v
    u = ds["u100"].values.astype(np.float32)
    v = ds["v100"].values.astype(np.float32)

    print("Processed CERRA file looks like :")
    print(ds)
    data = np.stack([u, v], axis=-1)

    return data, u, v


import tensorflow as tf


def main():
    """Compute the statistics of every ERA5 and CERRA file and write the CSV."""

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {len(gpus)} - {gpus[0].name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device = '/GPU:0'
            print(f"Using GPU: {device}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
            device = '/CPU:0'
    else:
        print("No GPU detected. Using CPU")
        device = '/CPU:0'

    n = np.array([2928, 2920, 2920, 2920, 2928])

    mean_u_ERA5 = np.array([0.477759,0.469028,0.470867,0.474134,0.471808])
    var_u_ERA5  = np.array([0.007080,0.006150,0.007030,0.007701,0.007184 ])

    mean_v_ERA5 = np.array([1.3739837408065796, 0.35431766510009766, 1.0236527919769287, 0.8157165050506592, 1.1657785177230835])
    var_v_ERA5  = np.array([30.7728271484375, 28.478857040405273, 28.621631622314453, 26.985576629638672, 28.294635772705078])

    mu_u_ERA5 = np.sum(n * mean_u_ERA5) / np.sum(n)
    varU_ERA5= np.sum(n * (var_u_ERA5 + (mean_u_ERA5 - mu_u_ERA5)**2)) / np.sum(n)

    mu_v_ERA5 = np.sum(n * mean_v_ERA5) / np.sum(n)
    varV_ERA5 = np.sum(n * (var_v_ERA5 + (mean_v_ERA5 - mu_v_ERA5)**2)) / np.sum(n)

    print(mu_u_ERA5, varU_ERA5, mu_v_ERA5, varV_ERA5)

    n = np.array([2928, 2920, 2920, 2920, 2928])
    mean_u_CERRA = np.array([0.492626,0.485647,0.486945, 0.489818, 0.487213])
    var_u_CERRA  = np.array([0.005802,0.005133,0.005822,0.006346,0.005806])

    mean_v_CERRA = np.array([1.5673739910125732, 0.5595088005065918, 1.178222417831421, 0.99039626121521, 1.3820536136627197])
    var_v_CERRA  = np.array([32.17461013793945, 30.093589782714844, 30.191408157348633, 28.786361694335938, 29.42989158630371])
    mu_u_CERRA = np.sum(n * mean_u_CERRA) / np.sum(n)
    varU_CERRA = np.sum(n * (var_u_CERRA + (mean_u_CERRA - mu_u_CERRA)**2)) / np.sum(n)

    mu_v_CERRA = np.sum(n * mean_v_CERRA) / np.sum(n)
    varV_CERRA = np.sum(n * (var_v_CERRA + (mean_v_CERRA - mu_v_CERRA)**2)) / np.sum(n)

    print("The mean and the stdev are for CERRA :")
    print(mu_u_CERRA, varU_CERRA, mu_v_CERRA, varV_CERRA)

    stats_rows = []

    era5_files = sorted(glob.glob(os.path.join(ERA5_INPUT_DIR, "*.grib")))
    cerra_files = sorted(glob.glob(os.path.join(CERRA_INPUT_DIR, "*.nc")))

    # Statistics of each ERA5 file
    for f in era5_files:

        year = extract_year(os.path.basename(f))
        data, u, v = process_era5_file(f)

        filename = f"ERA5_{year}.npy"

        stats = compute_stats_ERA5(u, v)

        stats_rows.append({
            "dataset": "ERA5",
            "year": year,
            **stats
        })

    # Statistics of each CERRA file
    for f in cerra_files:

        year = extract_year(os.path.basename(f))
        data, u, v = process_cerra_file(f)

        filename = f"CERRA_{year}.npy"

        stats = compute_stats_CERRA(u, v)

        stats_rows.append({
            "dataset": "CERRA",
            "year": year,
            **stats
        })

    # Write the statistics CSV
    df = pd.DataFrame(stats_rows)

    df = df[[
        "dataset", "year",
        "min_u", "max_u", "mean_u", "variance_u",
        "min_v", "max_v", "mean_v", "variance_v"
    ]]

    print(df)

    df.to_csv(CSV_LOG_PATH, index=False)

    print("\n Processing complete")
    print(f"CSV saved at: {CSV_LOG_PATH}")

    if ADD_COMBINED_STATS:
        era5_combined  = combine_stats(df, "ERA5")
        cerra_combined = combine_stats(df, "CERRA")
        print(era5_combined)
        print(cerra_combined)

        df = pd.concat([df, pd.DataFrame([era5_combined, cerra_combined])], ignore_index=True)

        df.to_csv(CSV_LOG_PATH, index=False)


if __name__ == "__main__":
    main()
