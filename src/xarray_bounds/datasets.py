import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds.options import OPTIONS

__all__ = ["tiny"]

rng = np.random.default_rng(12345)


def make_tiny_dataset(options=OPTIONS) -> xr.Dataset:
    bounds_dim = options["bounds_dim"]

    time_interval = pd.interval_range(
        start=pd.Timestamp("2000"), freq="MS", periods=5
    )
    time_bounds = xr.DataArray(
        data=np.stack([time_interval.left, time_interval.right], axis=-1),
        dims=["time", bounds_dim],
        name=f"time_{bounds_dim}",
    )
    time = xr.DataArray(
        data=pd.to_datetime(time_interval.left),
        dims="time",
        name="time",
        attrs={
            "standard_name": "time",
            "axis": "T",
            "bounds": time_bounds.name,
        },
    )
    lat_interval = pd.interval_range(start=50, freq=1, periods=3)
    lat_bounds = xr.DataArray(
        data=np.stack([lat_interval.left, lat_interval.right], axis=-1),
        dims=["lat", bounds_dim],
        name=f"lat_{bounds_dim}",
    )
    lat = xr.DataArray(
        data=lat_interval.mid,
        dims="lat",
        name="lat",
        attrs={
            "standard_name": "latitude",
            "axis": "Y",
            "bounds": lat_bounds.name,
        },
    )

    lon_interval = pd.interval_range(start=-2, freq=1, periods=4)
    lon_bounds = xr.DataArray(
        data=np.stack([lon_interval.left, lon_interval.right], axis=-1),
        dims=["lon", bounds_dim],
        name=f"lon_{bounds_dim}",
    )
    lon = xr.DataArray(
        data=lon_interval.right,
        dims="lon",
        name="lon",
        attrs={
            "standard_name": "longitude",
            "axis": "X",
            "bounds": lon_bounds.name,
        },
    )

    aux = xr.DataArray(
        data=["a", "b", "c", "d", "e"],
        dims=(time.name,),
        name="aux",
        attrs={"long_name": "Auxilliary coordinate"},
    )
    tas = xr.DataArray(
        data=rng.normal(size=(time.size, lat.size, lon.size)),
        dims=(time.name, lat.name, lon.name),
        name="tas",
        attrs={
            "long_name": "air temperature",
            "standard_name": "air_temperature",
            "units": "K",
        },
    )
    return xr.Dataset(
        data_vars={tas.name: tas},
        coords={
            time.name: time,
            lat.name: lat,
            lon.name: lon,
            time_bounds.name: time_bounds,
            lat_bounds.name: lat_bounds,
            lon_bounds.name: lon_bounds,
            aux.name: aux,
        },
        attrs={
            "name": "tiny_cf_dataset",
            "description": "A tiny synthetic CF-compliant dataset with independent coordinate variables and boundary variables.",
            "references": [
                "https://cfconventions.org/cf-conventions/cf-conventions.html#_independent_latitude_longitude_vertical_and_time_axes",
                "https://cfconventions.org/cf-conventions/cf-conventions.html#_data_representative_of_cells",
            ],
        },
    )


tiny = make_tiny_dataset()
