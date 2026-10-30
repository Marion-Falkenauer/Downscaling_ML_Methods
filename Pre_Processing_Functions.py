"""Preprocessing and plotting helpers for the ERA5 and CERRA datasets.

domain_crop, time_crop, wind_speed_to_uv: crop the datasets and convert wind speed and direction to u and v.
coarsening_method: strided downsampling of a sample.
plot_map, plot_comparison_HR_LR: map figures.

Author: Marion Falkenauer (with the help of Claude)
"""


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def domain_crop(df, lat_min, lat_max, lon_min, lon_max,variable_name="long"):
    """Crop a dataset to a latitude/longitude window.

    Args:
        df: xarray Dataset or DataArray.
        lat_min, lat_max: southern and northern bounds.
        lon_min, lon_max: western and eastern bounds.
        variable_name: "long" for dimensions latitude/longitude (latitude stored decreasing),
            "short" for dimensions lat/lon (latitude stored increasing).
    Returns:
        The cropped object.
    """

    print("Cropping to domain of interest...")
    if variable_name=="long":
        df_filtered = df.sel(
        latitude=slice(lat_max,lat_min),
        longitude=slice(lon_min,lon_max)
        )
    elif variable_name=="short":
        df_filtered = df.sel(
        lat=slice(lat_min,lat_max),
        lon=slice(lon_min,lon_max)
        )
    return df_filtered


def time_crop(df, start_time, end_time,time_var='time'):
    """Crop a dataset to a time range.

    Args:
        df: xarray Dataset or DataArray.
        start_time, end_time: bounds of the range.
        time_var: "time" (ERA5) or "valid_time" (CERRA).
    Returns:
        The cropped object.
    """
    if time_var=='time':
        df_cropped = df.sel(time=slice(start_time, end_time))
    elif time_var=='valid_time':
        df_cropped = df.sel(valid_time=slice(start_time, end_time))
    return df_cropped


def coarsening_method(sample, rf):
    """Downsample a sample by keeping every rf-th pixel.

    Args:
        sample: array (lat, lon, C).
        rf: reduction factor.
    Returns:
        Array (lat / rf, lon / rf, C).
    """
    return sample[::rf, ::rf, :]


def wind_speed_to_uv(xarray):
    """Convert wind speed and direction into u100 and v100.

    Args:
        xarray: Dataset with wind_speed and wind_direction (degrees, direction the wind blows from).
    Returns:
        Dataset with u100 and v100 in place of wind_speed and wind_direction.
    """
    print("Converting wind speed into u/v components")
    # Convert direction to radians
    wd_rad = np.deg2rad(xarray['wind_direction'])
    # Meteorological convention: direction the wind blows from
    xarray['u100'] = -xarray['wind_speed'] * np.sin(wd_rad)
    xarray['v100'] = -xarray['wind_speed'] * np.cos(wd_rad)
    xarray = xarray.drop_vars(['wind_speed', 'wind_direction'])
    return xarray


def plot_map(data, title, units='', colormap='viridis', levels=None,
             vmin=None, vmax=None):
    """Plot a 2-D field on a Lambert conformal map and save it as a PNG.

    The file is named plot-<title with underscores>.png.

    Args:
        data: xarray.DataArray with dimensions latitude and longitude.
        title: title of the plot.
        units: colourbar label.
        colormap: matplotlib colormap name.
        levels: optional contour levels.
        vmin, vmax: optional colour limits.
    """
    proj = ccrs.LambertConformal(
        central_longitude=2.5,
        central_latitude=53.6,
        standard_parallels=(30, 60)
    )

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})

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

    # Save as PNG, named from the title
    save_title = "plot-" + title.replace(" ", "_")
    plt.savefig(f"{save_title}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot as {save_title}.png")


def plot_comparison_HR_LR(ds_HR,ds_LR,var="u100",time_idx=5,titles=("High resolution", "Low resolution"),cmap="viridis",extent=[-15, 35, 30, 70],savepath=None):
    """Plot a HR and a LR field side by side on maps with a shared colour scale.

    Args:
        ds_HR, ds_LR: datasets containing var.
        var: variable name.
        time_idx: time index to plot.
        titles: titles of the two panels.
        cmap: colormap name.
        extent: [lon_min, lon_max, lat_min, lat_max], or None.
        savepath: file to save the figure to (the figure is then closed).
    """

    # Compute common color limits (Dask-safe)
    vmin = min(
        ds_HR[var].isel(time=time_idx).min().compute().item(),
        ds_LR[var].isel(time=time_idx).min().compute().item()
    )
    vmax = max(
        ds_HR[var].isel(time=time_idx).max().compute().item(),
        ds_LR[var].isel(time=time_idx).max().compute().item()
    )

    # Create Cartopy axes
    fig, axes = plt.subplots(1, 2,figsize=(12, 4),subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
    )

    # Plot HR
    p1 = ds_HR[var].isel(time=time_idx).plot(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False
    )
    axes[0].set_title(titles[0])

    # Plot LR
    ds_LR[var].isel(time=time_idx).plot(
        ax=axes[1],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False
    )
    axes[1].set_title(titles[1])

    # Add map features
    for ax in axes:
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black", zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=0)

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Shared colorbar
    cbar = fig.colorbar(p1, ax=axes, orientation="vertical", shrink=0.9)
    cbar.set_label(var)

    # Save or show
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        plt.close(fig)
