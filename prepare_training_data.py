"""Prepare the training, validation and test memmaps for the downscaling experiments.

ERA5 (LR input) and CERRA (HR target) wind at 100 m are cropped, converted to u and v, stacked,
and written as float16 memmaps. Statistics (min, max, mean, variance) are computed on the raw
float32 data before the cast, per split, and saved in CSV files.

Storage layout
--------------
Validation (2023) and test (2024) are shared by all training cases and written once:

    Data_Shared/
        val/   ERA5_val.npy   ERA5_val_shape.txt   CERRA_val.npy   CERRA_val_shape.txt
        test/  ERA5_test.npy  ERA5_test_shape.txt  CERRA_test.npy  CERRA_test_shape.txt
        coords/  lat_LR.npy  lon_LR.npy  lat_HR.npy  lon_HR.npy
        statistics_shared.csv

Only the training data depends on the case:

    Data_<case>/
        train/  ERA5_train.npy  ERA5_train_shape.txt  CERRA_train.npy  CERRA_train_shape.txt
        statistics_<case>.csv

Each *_shape.txt holds the comma-separated shape on the first line (e.g. 2920,320,400,2) and the
dtype on the second line (float16), which is read by a sidcar to adapt the correct shape of the tensor to the algorithm downstream.

Preprocessing
-------------
    ERA5: cfgrib, time crop, domain crop, 3-hourly samples, u100 and v100.
    CERRA: h5netcdf, time crop, domain crop, wind speed and direction to u and v, latitude flip.
    Both: stack to (T, lat, lon, 2). LR: sequence_encoding. HR: HR_sequence_slicing.
    UV_COMP False keeps only the u channel.

Splits
------
    Test: 2024-01-01 to 2024-12-31.
    Validation: 2023-01-07 to 2023-12-24 (margins are taken to avoid autocorrelation with test and training).
    Training: from the start year of the case to 2022-12-31.

Configuration (module constants):
    CASE_START_YEARS: first training year of each case; CASES selects the cases to run.
    CONVERTING_DATA: master switch, False does nothing.
    COMPUTE_STATISTICS: write the statistics CSV files.
    FORCE_SHARED: rewrite the shared validation and test data even if they exist.
    N_BEFORE, N_AFTER, UV_COMP: LR input layout.
    BASE_DIR, OUTPUT_ROOT: input data root and output root, to adapt to the local file system.

Reload a memmap with loading_uv_normalize(npy_path, shape_path, csv_path, dataset, mode); always
pass the CSV of the training case so that validation and test use the training constants.

Author: Marion Falkenauer (with the help of Claude)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:

        tf.config.experimental.set_memory_growth(gpus[0], True)

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}")
        print(f"Using GPU: {gpus[0].name}")
        print(f"Logical GPUs: {logical_gpus}")

    except RuntimeError as e:
        print(e)

else:
    print("No GPU found, using CPU.")


import re
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from Pre_Processing_Functions import time_crop, domain_crop, wind_speed_to_uv
from Tensor_Adaptation import sequence_encoding,HR_sequence_slicing

BASE_DIR = "insert base directory (ex: RP_Codes)"
DATA_DIR  = os.path.join(BASE_DIR, "Data")

ERA5_INPUT_DIR  = os.path.join(DATA_DIR, "ERA5")
CERRA_INPUT_DIR = os.path.join(DATA_DIR, "CERRA")

OUTPUT_ROOT = "insert base directory (ex: RP_Codes/Results)"

LON_COORDS       = [-5,    14.75]
LAT_COORDS       = [ 44,   59.75]
LON_COORDS_CERRA = [-5,    14.95]
LAT_COORDS_CERRA = [ 44,   59.95]

N_BEFORE = 1
N_AFTER  = 0
UV_COMP  = True

TARGET_HEIGHT = 320
TARGET_WIDTH  = 400

CASE_START_YEARS = {
    "2year": 2021,
}
CASES = list(CASE_START_YEARS)

VAL_START  = "2023-01-07"
VAL_END    = "2023-12-24"
TEST_START = "2024-01-01"
TEST_END   = "2024-12-31"
TRAIN_END  = "2022-12-31"

CONVERTING_DATA    = True
COMPUTE_STATISTICS = True
FORCE_SHARED       = False

STORAGE_DTYPE = np.float16

ALLOWED_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

WIND_VARS = ["100u", "100v"]

_SHARED_SENTINEL = ".shared_complete"


class SectionTimer:
    """Wall-clock timer for named sections of a run."""

    def __init__(self):
        """Create empty dictionaries of start times and durations."""
        self._starts = {}
        self.times   = {}

    def start(self, name):
        """Start the timer of a section.

        Args:
            name: section label.
        """
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        """Stop a section and store its duration in seconds in self.times.

        Args:
            name: section label given to start.
        Returns:
            Duration in seconds.
        """
        elapsed = time.perf_counter() - self._starts[name]
        self.times[name] = elapsed
        return elapsed


def load_era5_variables(files, variables, level_type=None, file_type="cfgrib"):
    """Load ERA5 or CERRA files with Dask chunks along time and concatenate them.

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
        print("  Loading {} ...".format(os.path.basename(f)))
        if file_type == "cfgrib":
            ds = xr.open_dataset(
                f,
                engine="cfgrib",
                backend_kwargs=backend_kwargs,
                chunks={"time": 100},
            )
        elif file_type in ["netcdf4", "h5netcdf"]:
            ds = xr.open_dataset(f, engine=file_type, chunks={"time": 100})
        else:
            raise ValueError("Unsupported file_type: {!r}".format(file_type))
        datasets.append(ds)

    return xr.concat(datasets, dim="time").sortby("time")


def preprocess_era5(ds, time_start, time_end):
    """Crop an ERA5 dataset and extract the wind components.

    Steps: time crop, domain crop, 3-hourly samples. Arrays stay float32 until they are written.

    Args:
        ds: ERA5 dataset.
        time_start, time_end: time range.
    Returns:
        (u, v, lat, lon): float32 arrays (T, lat, lon) of u100 and v100 and the coordinate vectors.
    """
    ds   = time_crop(ds, time_start, time_end, time_var="time")
    ds   = domain_crop(ds,
                       LAT_COORDS[0], LAT_COORDS[1],
                       LON_COORDS[0], LON_COORDS[1],
                       variable_name="long")
    mask = ds["time"].dt.hour.isin(ALLOWED_HOURS)
    ds   = ds.isel(time=mask)
    u    = ds["u100"].values.astype(np.float32)
    v    = ds["v100"].values.astype(np.float32)
    return u, v, ds["latitude"].values, ds["longitude"].values


def preprocess_cerra(ds, time_start, time_end):
    """Crop a CERRA dataset and extract the wind components.

    Steps: time crop, domain crop, wind speed and direction to u and v, latitude flip.

    Args:
        ds: CERRA dataset.
        time_start, time_end: time range.
    Returns:
        (u, v, lat, lon): float32 arrays (T, lat, lon) of u100 and v100 and the coordinate vectors.
    """
    ds = time_crop(ds, time_start, time_end, time_var="time")
    ds = domain_crop(ds,
                     LAT_COORDS_CERRA[0], LAT_COORDS_CERRA[1],
                     LON_COORDS_CERRA[0], LON_COORDS_CERRA[1],
                     variable_name="short")
    ds = wind_speed_to_uv(ds)
    ds = ds.sel(lat=slice(None, None, -1))
    u  = ds["u100"].values.astype(np.float32)
    v  = ds["v100"].values.astype(np.float32)
    return u, v, ds["lat"].values, ds["lon"].values


def extract_year(filename):
    """Return the first four-digit number found in a file name, or None."""
    match = re.search(r"\d{4}", os.path.basename(filename))
    return int(match.group()) if match else None


def files_in_year_range(file_list, year_start, year_end):
    """Select the files whose name contains a year within a range.

    Args:
        file_list: file paths.
        year_start, year_end: inclusive year range.
    Returns:
        Sorted list of the matching files.
    """
    result = []
    for f in file_list:
        y = extract_year(f)
        if y is not None and year_start <= y <= year_end:
            result.append(f)
    return sorted(result)


def load_and_preprocess_era5(file_list, time_start, time_end):
    """Load ERA5 files and apply preprocess_era5.

    Args:
        file_list: GRIB files.
        time_start, time_end: time range.
    Returns:
        (u, v, lat, lon) as in preprocess_era5.
    """
    ds = load_era5_variables(file_list, WIND_VARS, file_type="cfgrib")
    return preprocess_era5(ds, time_start, time_end)


def load_and_preprocess_cerra(file_list, time_start, time_end):
    """Load CERRA files and apply preprocess_cerra.

    Args:
        file_list: NetCDF files.
        time_start, time_end: time range.
    Returns:
        (u, v, lat, lon) as in preprocess_cerra.
    """
    ds = load_era5_variables(file_list, WIND_VARS, file_type="h5netcdf")
    return preprocess_cerra(ds, time_start, time_end)


def compute_raw_stats(u, v, dataset, split):
    """Statistics of the raw float32 u and v components of a split.

    Args:
        u, v: arrays (T, lat, lon).
        dataset: "ERA5" or "CERRA".
        split: "train", "val" or "test".
    Returns:
        Dictionary with dataset, split, n_timesteps and the min, max, mean, variance and std of u and v.
    """
    return {
        "dataset":     dataset,
        "split":       split,
        "n_timesteps": int(u.shape[0]),
        "n_pixels":    int(u.size),
        "min_u":       float(np.min(u)),
        "max_u":       float(np.max(u)),
        "mean_u":      float(np.mean(u)),
        "var_u":       float(np.var(u)),
        "std_u":       float(np.std(u)),
        "min_v":       float(np.min(v)),
        "max_v":       float(np.max(v)),
        "mean_v":      float(np.mean(v)),
        "var_v":       float(np.var(v)),
        "std_v":       float(np.std(v)),
    }


def write_memmap(array, path):
    """Write an array as a float16 memmap with a shape and dtype sidecar file.

    The sidecar (same path with _shape.txt) lets readers recover shape and dtype. float16 halves the
    storage and is sufficient for wind values bounded by about 50 m/s.

    Args:
        array: array to store.
        path: .npy file to create.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr16 = array.astype(STORAGE_DTYPE)
    fp    = np.memmap(path, dtype=STORAGE_DTYPE, mode="w+", shape=arr16.shape)
    fp[:] = arr16[:]
    fp.flush()
    del fp  # close / flush to disk
    _write_sidecar(path, arr16.shape, STORAGE_DTYPE)
    print("    -> saved  {}  shape={}  dtype={}".format(
        path, arr16.shape, STORAGE_DTYPE.__name__))


def _write_sidecar(npy_path, shape, dtype):
    """Write the shape and dtype of a memmap to its _shape.txt sidecar."""
    meta = npy_path.replace(".npy", "_shape.txt")
    with open(meta, "w") as fh:
        fh.write(",".join(str(s) for s in shape) + "\n")
        fh.write(np.dtype(dtype).name + "\n")


def _read_sidecar(shape_path):
    """Read the sidecar of a memmap.

    Args:
        shape_path: _shape.txt file.
    Returns:
        (shape, dtype); dtype defaults to float32 when absent.
    """
    lines = open(shape_path).read().strip().splitlines()
    shape = tuple(int(s) for s in lines[0].split(","))
    dtype = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
    return shape, dtype


def build_shared_splits(timer):
    """Write the shared validation and test memmaps, coordinates and statistics.

    The work is skipped when Data_Shared/.shared_complete exists, unless FORCE_SHARED is True.

    Args:
        timer: SectionTimer collecting the section times.
    Returns:
        Path of the Data_Shared folder.
    """
    shared_dir   = os.path.join(OUTPUT_ROOT, "Data_Shared2t")
    sentinel     = os.path.join(shared_dir, _SHARED_SENTINEL)
    val_dir      = os.path.join(shared_dir, "val")
    test_dir     = os.path.join(shared_dir, "test")
    coords_dir   = os.path.join(shared_dir, "coords")
    csv_path     = os.path.join(shared_dir, "statistics2t_shared.csv")

    if os.path.exists(sentinel) and not FORCE_SHARED:
        print("Shared val/test already exist ({}).".format(shared_dir))
        print("  Set FORCE_SHARED=True to overwrite.")
        return shared_dir

    print("\n" + "=" * 62)
    print("  Building SHARED val / test splits")
    print("=" * 62)

    for d in [val_dir, test_dir, coords_dir]:
        os.makedirs(d, exist_ok=True)

    # Gather files
    all_era5  = sorted(glob.glob(os.path.join(ERA5_INPUT_DIR,  "*.grib")))
    all_cerra = sorted(glob.glob(os.path.join(CERRA_INPUT_DIR, "*.nc")))

    val_era5   = files_in_year_range(all_era5,  2023, 2023)
    val_cerra  = files_in_year_range(all_cerra, 2023, 2023)
    test_era5  = files_in_year_range(all_era5,  2024, 2024)
    test_cerra = files_in_year_range(all_cerra, 2024, 2024)

    for label, lst in [
        ("ERA5  val",  val_era5),
        ("CERRA val",  val_cerra),
        ("ERA5  test", test_era5),
        ("CERRA test", test_cerra),
    ]:
        if not lst:
            raise FileNotFoundError(
                "No files found for {!r}. "
                "Check ERA5_INPUT_DIR / CERRA_INPUT_DIR.".format(label))
        print("  {}: {} file(s) -> {}".format(
            label, len(lst), [os.path.basename(f) for f in lst]))

    # Load & preprocess
    timer.start("shared_preprocessing")

    print("\n  Loading VAL data ...")
    vu_era5,  vv_era5,  lat_lr, lon_lr = load_and_preprocess_era5(
        val_era5, VAL_START, VAL_END)
    vu_cerra, vv_cerra, lat_hr, lon_hr = load_and_preprocess_cerra(
        val_cerra, VAL_START, VAL_END)

    print("\n  Loading TEST data ...")
    xu_era5,  xv_era5,  _, _ = load_and_preprocess_era5(
        test_era5, TEST_START, TEST_END)
    xu_cerra, xv_cerra, _, _ = load_and_preprocess_cerra(
        test_cerra, TEST_START, TEST_END)

    timer.stop("shared_preprocessing")

    # Sanity checks
    for sp, n_e, n_c in [
        ("val",  vu_era5.shape[0], vu_cerra.shape[0]),
        ("test", xu_era5.shape[0], xu_cerra.shape[0]),
    ]:
        if n_e != n_c:
            raise ValueError(
                "[shared] Timestep mismatch in {!r}: "
                "ERA5={}, CERRA={}.".format(sp, n_e, n_c))
        print("  [{}] timestep alignment OK -- {} steps".format(sp, n_e))

    ny, nx = vu_cerra.shape[1], vu_cerra.shape[2]
    if ny < TARGET_HEIGHT or nx < TARGET_WIDTH:
        raise ValueError(
            "[shared] CERRA grid too small: (height={}, width={}), "
            "need at least ({}, {}).".format(ny, nx, TARGET_HEIGHT, TARGET_WIDTH))
    print("  Spatial size OK -- HR (height={}, width={})  "
          "LR (height={}, width={})".format(
              ny, nx, vu_era5.shape[1], vu_era5.shape[2]))

    # Statistics (float32, before any cast)
    if COMPUTE_STATISTICS:
        print("\n  Computing RAW statistics for shared splits ...")
        stats_rows = []
        for u, v, ds_label, sp in [
            (vu_era5,  vv_era5,  "ERA5",  "val"),
            (vu_cerra, vv_cerra, "CERRA", "val"),
            (xu_era5,  xv_era5,  "ERA5",  "test"),
            (xu_cerra, xv_cerra, "CERRA", "test"),
        ]:
            stats_rows.append(compute_raw_stats(u, v, ds_label, sp))

        df_stats = pd.DataFrame(stats_rows)[[
            "dataset", "split", "n_timesteps", "n_pixels",
            "min_u", "max_u", "mean_u", "var_u", "std_u",
            "min_v", "max_v", "mean_v", "var_v", "std_v",
        ]]
        df_stats.to_csv(csv_path, index=False)
        print("  Shared statistics CSV -> {}".format(csv_path))
        print(df_stats.to_string(index=False))

    # Stack channels: (T, lat, lon, 2)
    lr_val  = np.stack([vu_era5,  vv_era5],  axis=-1)
    lr_test = np.stack([xu_era5,  xv_era5],  axis=-1)
    hr_val  = np.stack([vu_cerra, vv_cerra], axis=-1)
    hr_test = np.stack([xu_cerra, xv_cerra], axis=-1)

    # Sequence encoding
    print("\n  Applying sequence_encoding / HR_sequence_slicing ...")
    lr_val  = sequence_encoding(lr_val,  N_BEFORE, N_AFTER, keep_channels=True)
    lr_test = sequence_encoding(lr_test, N_BEFORE, N_AFTER, keep_channels=True)
    for tag, arr in [("lr_val", lr_val), ("lr_test", lr_test)]:
        if arr.ndim == 5:
            T_, H_, W_, nf, nc = arr.shape
            arr = arr.reshape(T_, H_, W_, nf * nc)
            if tag == "lr_val":
                lr_val  = arr
            else:
                lr_test = arr
    hr_val  = HR_sequence_slicing(hr_val,  N_BEFORE, N_AFTER)
    hr_test = HR_sequence_slicing(hr_test, N_BEFORE, N_AFTER)

    # UV_COMP trim
    if not UV_COMP:
        lr_val  = lr_val[...,  0:1]
        lr_test = lr_test[..., 0:1]
        hr_val  = hr_val[...,  0:1]
        hr_test = hr_test[..., 0:1]

    print("\n  Post-sequence shapes:")
    print("    LR  val={}  test={}".format(lr_val.shape, lr_test.shape))
    print("    HR  val={}  test={}".format(hr_val.shape, hr_test.shape))

    # Post-sequence timestep alignment check
    for sp, n_lr, n_hr in [
        ("val",  lr_val.shape[0],  hr_val.shape[0]),
        ("test", lr_test.shape[0], hr_test.shape[0]),
    ]:
        if n_lr != n_hr:
            raise ValueError(
                "[shared] Post-sequence mismatch in {!r}: "
                "LR={}, HR={}.".format(sp, n_lr, n_hr))
        print("  [{}] post-sequence alignment OK -- {} samples".format(sp, n_lr))

    # Write float16 memmaps
    print("\n  Writing shared float16 memmap arrays ...")
    for sp, lr_arr, hr_arr, out_dir in [
        ("val",  lr_val,  hr_val,  val_dir),
        ("test", lr_test, hr_test, test_dir),
    ]:
        write_memmap(lr_arr, os.path.join(out_dir, "ERA5_{}.npy".format(sp)))
        write_memmap(hr_arr, os.path.join(out_dir, "CERRA_{}.npy".format(sp)))

    # Coordinate arrays
    print("\n  Saving coordinate arrays ...")
    np.savetxt(os.path.join(coords_dir, "lat_LR.npy"), lat_lr)
    np.savetxt(os.path.join(coords_dir, "lon_LR.npy"), lon_lr)
    np.savetxt(os.path.join(coords_dir, "lat_HR.npy"), lat_hr)
    np.savetxt(os.path.join(coords_dir, "lon_HR.npy"), lon_hr)

    # Sentinel
    with open(sentinel, "w") as fh:
        fh.write(datetime.now().isoformat())
    print("\n  Shared splits complete -> {}".format(shared_dir))
    return shared_dir


def process_case(case_name, train_start_year, timer):
    """Write the training memmaps and statistics of one case.

    Steps: gather the training files, load and preprocess them, check time and space alignment,
    compute the raw statistics, stack u and v, apply sequence encoding to LR and slicing to HR,
    optionally keep only u, check alignment again and write the float16 memmaps.

    Args:
        case_name: key of CASE_START_YEARS.
        train_start_year: first training year.
        timer: SectionTimer collecting the section times.
    """
    print("\n" + "=" * 62)
    print("  CASE : {}  (training {}-2022)".format(case_name, train_start_year))
    print("=" * 62)

    # Output directories
    case_dir  = os.path.join(OUTPUT_ROOT, "Data2t_{}".format(case_name))
    train_dir = os.path.join(case_dir, "train2t")
    os.makedirs(train_dir, exist_ok=True)

    csv_path = os.path.join(case_dir, "statistics2t_{}.csv".format(case_name))

    # Gather training files
    all_era5  = sorted(glob.glob(os.path.join(ERA5_INPUT_DIR,  "*.grib")))
    all_cerra = sorted(glob.glob(os.path.join(CERRA_INPUT_DIR, "*.nc")))

    train_era5  = files_in_year_range(all_era5,  train_start_year, 2022)
    train_cerra = files_in_year_range(all_cerra, train_start_year, 2022)

    for label, lst in [
        ("ERA5  train [{}]".format(case_name), train_era5),
        ("CERRA train [{}]".format(case_name), train_cerra),
    ]:
        if not lst:
            raise FileNotFoundError(
                "No files found for {!r}. "
                "Check ERA5_INPUT_DIR / CERRA_INPUT_DIR.".format(label))
        print("  {}: {} file(s) -> {}".format(
            label, len(lst), [os.path.basename(f) for f in lst]))

    # Load & preprocess training split
    timer.start("preprocessing_{}".format(case_name))

    print("\n  [{}] Loading TRAINING data ...".format(case_name))
    tu_era5,  tv_era5,  lat_lr, lon_lr = load_and_preprocess_era5(
        train_era5, "{}-01-01".format(train_start_year), TRAIN_END)
    tu_cerra, tv_cerra, lat_hr, lon_hr = load_and_preprocess_cerra(
        train_cerra, "{}-01-01".format(train_start_year), TRAIN_END)

    elapsed = timer.stop("preprocessing_{}".format(case_name))
    print("  Training preprocessing done in {:.1f}s".format(elapsed))

    # Sanity check: timestep alignment
    if tu_era5.shape[0] != tu_cerra.shape[0]:
        raise ValueError(
            "[{}] Timestep mismatch in train: "
            "ERA5={}, CERRA={}.".format(
                case_name, tu_era5.shape[0], tu_cerra.shape[0]))
    print("  [train] timestep alignment OK -- {} steps".format(tu_era5.shape[0]))

    # Sanity check: spatial size guard (HEIGHT=lat, WIDTH=lon)
    ny, nx = tu_cerra.shape[1], tu_cerra.shape[2]
    if ny < TARGET_HEIGHT or nx < TARGET_WIDTH:
        raise ValueError(
            "[{}] CERRA grid too small: (height={}, width={}), "
            "need at least ({}, {}).".format(
                case_name, ny, nx, TARGET_HEIGHT, TARGET_WIDTH))
    print("  Spatial size OK -- "
          "HR (height={}, width={})  LR (height={}, width={})".format(
              ny, nx, tu_era5.shape[1], tu_era5.shape[2]))

    # RAW statistics at float32 (before cast)
    if COMPUTE_STATISTICS:
        print("\n  Computing RAW statistics ...")
        stats_rows = [
            compute_raw_stats(tu_era5,  tv_era5,  "ERA5",  "train"),
            compute_raw_stats(tu_cerra, tv_cerra, "CERRA", "train"),
        ]
        df_stats = pd.DataFrame(stats_rows)[[
            "dataset", "split", "n_timesteps", "n_pixels",
            "min_u", "max_u", "mean_u", "var_u", "std_u",
            "min_v", "max_v", "mean_v", "var_v", "std_v",
        ]]
        df_stats.to_csv(csv_path, index=False)
        print("  Statistics CSV -> {}".format(csv_path))
        print(df_stats.to_string(index=False))

    # Stack channels: (T, lat, lon, 2)
    lr_train = np.stack([tu_era5,  tv_era5],  axis=-1)
    hr_train = np.stack([tu_cerra, tv_cerra], axis=-1)

    print("\n  Pre-sequence shapes:")
    print("    LR train={}   HR train={}".format(lr_train.shape, hr_train.shape))

    # Sequence encoding and HR slicing
    print("  Applying sequence_encoding / HR_sequence_slicing ...")
    lr_train = sequence_encoding(lr_train, N_BEFORE, N_AFTER, keep_channels=True)
    if lr_train.ndim == 5:
        T_, H_, W_, nf, nc = lr_train.shape
        lr_train = lr_train.reshape(T_, H_, W_, nf * nc)

    hr_train = HR_sequence_slicing(hr_train, N_BEFORE, N_AFTER)

    # UV_COMP trim
    if not UV_COMP:
        lr_train = lr_train[..., 0:1]
        hr_train = hr_train[..., 0:1]

    print("  Post-sequence shapes:")
    print("    LR train={}   HR train={}".format(lr_train.shape, hr_train.shape))

    # Post-sequence alignment check
    if lr_train.shape[0] != hr_train.shape[0]:
        raise ValueError(
            "[{}] Post-sequence mismatch in train: "
            "LR={}, HR={}.".format(
                case_name, lr_train.shape[0], hr_train.shape[0]))
    print("  [train] post-sequence alignment OK -- {} samples".format(
        lr_train.shape[0]))

    # Write float16 memmap (training only)
    print("\n  Writing float16 memmap arrays ...")
    write_memmap(lr_train, os.path.join(train_dir, "ERA5_train.npy"))
    write_memmap(hr_train, os.path.join(train_dir, "CERRA_train.npy"))

    # Helpful reload reminder
    shared_dir = os.path.join(OUTPUT_ROOT, "Data_Shared")
    print("\n  Case [{}] complete -> {}".format(case_name, case_dir))
    print("  Shared val/test  -> {}".format(shared_dir))
    print("  Reload example:")
    print("    from prepare_training_data import loading_uv_normalize")
    print("    X_train = loading_uv_normalize(")
    print("        npy_path   = '{}/ERA5_train.npy',".format(train_dir))
    print("        shape_path = '{}/ERA5_train_shape.txt',".format(train_dir))
    print("        csv_path   = '{}',".format(csv_path))
    print("        dataset    = 'ERA5', mode='minmax')")
    print("    X_val = loading_uv_normalize(")
    print("        npy_path   = '{}/val/ERA5_val.npy',".format(shared_dir))
    print("        shape_path = '{}/val/ERA5_val_shape.txt',".format(shared_dir))
    print("        csv_path   = '{}',  # training stats!".format(csv_path))
    print("        dataset    = 'ERA5', mode='minmax')")


def main():
    """Build the shared splits, then process every case in CASES."""
    run_start = datetime.now()
    print("prepare_training_data.py")
    print("Run started        : {}".format(run_start.isoformat(timespec="seconds")))
    print("Cases              : {}".format(CASES))
    print("N_BEFORE / N_AFTER : {} / {}".format(N_BEFORE, N_AFTER))
    print("UV_COMP            : {}".format(UV_COMP))
    print("STORAGE_DTYPE      : {}".format(STORAGE_DTYPE.__name__))
    print("COMPUTE_STATISTICS : {}".format(COMPUTE_STATISTICS))
    print("FORCE_SHARED       : {}".format(FORCE_SHARED))
    print("CONVERTING_DATA    : {}".format(CONVERTING_DATA))

    if not CONVERTING_DATA:
        print("\nCONVERTING_DATA=False -- nothing to do.")
        return

    timer = SectionTimer()
    total_start = time.perf_counter()

    # Step A: write shared val/test once
    build_shared_splits(timer)

    # Step B: write per-case training data
    for case_name in CASES:
        if case_name not in CASE_START_YEARS:
            raise ValueError(
                "Unknown case {!r}. Valid: {}".format(
                    case_name, list(CASE_START_YEARS)))
        process_case(case_name, CASE_START_YEARS[case_name], timer)

    total_elapsed = time.perf_counter() - total_start
    print("\nAll cases complete in {:.1f} min.".format(total_elapsed / 60.0))
    print("Section times:")
    for name, t in timer.times.items():
        print("  {}: {:.1f}s".format(name, t))

    # Disk usage summary
    print("\nEstimated on-disk sizes:")
    for folder in (
        [os.path.join(OUTPUT_ROOT, "Data_Shared")]
        + [os.path.join(OUTPUT_ROOT, "Data_{}".format(c)) for c in CASES]
    ):
        if os.path.isdir(folder):
            total_bytes = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fnames in os.walk(folder)
                for f in fnames
            )
            print("  {:40s}  {:6.1f} GB".format(
                os.path.basename(folder), total_bytes / 1e9))


if __name__ == "__main__":
    main()


def loading_uv_normalize(npy_path, shape_path, csv_path, dataset, mode="minmax"):
    """Load a float16 memmap and normalise u and v with the statistics of the training split.

    Args:
        npy_path: memmap file (training, validation or test).
        shape_path: matching _shape.txt sidecar.
        csv_path: statistics CSV of the training case, also for validation and test data.
        dataset: "ERA5" or "CERRA".
        mode: "minmax" gives values in [0, 1], "zscore" gives zero mean and unit variance.
    Returns:
        float32 array with the shape of the memmap; even channels are u and odd channels are v.
    """
    valid_modes = ("minmax", "zscore")
    if mode not in valid_modes:
        raise ValueError(
            "Unknown mode {!r}. Choose from {}.".format(mode, valid_modes))

    # Read shape AND dtype from the two-line sidecar
    shape, dtype = _read_sidecar(shape_path)

    # Open the float16 memmap read-only, then immediately cast to float32
    # for all subsequent arithmetic (float16 arithmetic causes precision loss)
    raw      = np.memmap(npy_path, dtype=dtype, mode="r", shape=shape)
    data_f32 = np.array(raw, dtype=np.float32)  # full float32 copy in RAM

    # Read normalisation constants from the TRAINING row
    df        = pd.read_csv(csv_path)
    train_row = df[(df["dataset"] == dataset) & (df["split"] == "train")]
    if train_row.empty:
        raise ValueError(
            "No 'train' row found for dataset={!r} in {}.".format(
                dataset, csv_path))
    row = train_row.iloc[0]

    n_channels = shape[-1]

    if mode == "minmax":
        min_u, max_u = float(row["min_u"]), float(row["max_u"])
        min_v, max_v = float(row["min_v"]), float(row["max_v"])
        _check_range("u", min_u, max_u)
        _check_range("v", min_v, max_v)
        for c in range(n_channels):
            if c % 2 == 0:
                data_f32[..., c] = (data_f32[..., c] - min_u) / (max_u - min_u)
            else:
                data_f32[..., c] = (data_f32[..., c] - min_v) / (max_v - min_v)

    elif mode == "zscore":
        mean_u, std_u = float(row["mean_u"]), float(row["std_u"])
        mean_v, std_v = float(row["mean_v"]), float(row["std_v"])
        _check_std("u", std_u)
        _check_std("v", std_v)
        for c in range(n_channels):
            if c % 2 == 0:
                data_f32[..., c] = (data_f32[..., c] - mean_u) / std_u
            else:
                data_f32[..., c] = (data_f32[..., c] - mean_v) / std_v
    print("loading_uv_normalize: {} shape={} dtype={}->{} dataset={} mode={}".format(
        os.path.basename(npy_path), shape, dtype, "float32", dataset, mode))

    return data_f32


def _check_range(component, vmin, vmax):
    """Raise ValueError if the min-max range of a component is zero."""
    if vmax - vmin == 0.0:
        raise ValueError(
            "Degenerate normalisation range for {!r}: "
            "min == max == {}.".format(component, vmin))


def _check_std(component, std):
    """Raise ValueError if the standard deviation of a component is zero."""
    if std == 0.0:
        raise ValueError(
            "Zero std for {!r}. Check the statistics CSV.".format(component))
