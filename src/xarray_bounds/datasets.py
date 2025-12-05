import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds import options

BOUNDS_DIM = options.OPTIONS['bounds_dim']

# --------------------------------------------------- Simple 1D coordinate axes
time = xr.DataArray(
    data=pd.to_datetime(
        [
            '2000-01-01T12:00',
            '2000-01-02T12:00',
            '2000-01-03T12:00',
            '2000-01-04T12:00',
        ]
    ),
    dims=('time',),
    name='time',
    attrs={'standard_name': 'time', 'axis': 'T'},
)

time_bounds = xr.DataArray(
    data=[
        pd.DatetimeIndex(['2000-01-01', '2000-01-02']),
        pd.DatetimeIndex(['2000-01-02', '2000-01-03']),
        pd.DatetimeIndex(['2000-01-03', '2000-01-04']),
        pd.DatetimeIndex(['2000-01-04', '2000-01-05']),
    ],
    dims=('time', BOUNDS_DIM),
    coords={'time': time},
    name=f'time_{BOUNDS_DIM}',
)

lat_1d = xr.DataArray(
    data=[90, 100, 110],
    dims=('lat',),
    name='lat',
    attrs={'standard_name': 'latitude', 'axis': 'Y', 'units': 'degrees_north'},
)

lat_1d_bounds = xr.DataArray(
    data=[
        [85, 95],
        [95, 105],
        [105, 115],
    ],
    dims=('lat', BOUNDS_DIM),
    coords={'lat': lat_1d},
    name=f'lat_{BOUNDS_DIM}',
)

lon_1d = xr.DataArray(
    data=[180, 190],
    dims=('lon',),
    name='lon',
    attrs={'standard_name': 'longitude', 'axis': 'X', 'units': 'degrees_east'},
)

lon_1d_bounds = xr.DataArray(
    data=[
        [175, 185],
        [185, 195],
    ],
    dims=('lon', BOUNDS_DIM),
    coords={'lon': lon_1d},
    name=f'lon_{BOUNDS_DIM}',
)

aux = xr.DataArray(
    data=['a', 'b', 'c', 'd'],
    dims=('time',),
    name='aux',
    attrs={'long_name': 'auxiliary variable'},
)

temperature_1d = xr.DataArray(
    data=np.random.rand(4, 3, 2) * 300,
    dims=('time', 'lat', 'lon'),
    name='temperature',
    attrs={'standard_name': 'air_temperature', 'units': 'K'},
)

simple = xr.Dataset(
    data_vars={'temperature': temperature_1d},
    coords={
        'time': time,
        'lat': lat_1d,
        'lon': lon_1d,
        'aux': aux,
    },
    attrs={
        'description': 'A synthetic CF-compliant dataset with independent coordinate variables.',
        'references': 'https://cfconventions.org/cf-conventions/cf-conventions.html#_independent_latitude_longitude_vertical_and_time_axes',
        'see_also': 'xarray_bounds.datasets.simple_bounds',
    },
)

simple_bounds = simple.copy(deep=False)
simple_bounds = simple_bounds.assign_coords(
    {
        f'time_{BOUNDS_DIM}': time_bounds,
        f'lat_{BOUNDS_DIM}': lat_1d_bounds,
        f'lon_{BOUNDS_DIM}': lon_1d_bounds,
    }
).assign_attrs(
    {
        'description': simple.attrs['description'].rstrip('.')
        + ' and boundary coordinate variables added.',
    }
)
simple_bounds['time'].attrs['bounds'] = f'time_{BOUNDS_DIM}'
simple_bounds['lat'].attrs['bounds'] = f'lat_{BOUNDS_DIM}'
simple_bounds['lon'].attrs['bounds'] = f'lon_{BOUNDS_DIM}'


# --------------------------------------------------- Simple 2D coordinate axes
xc = xr.DataArray(
    data=[1000.0, 1500.0],
    dims=('xc',),
    attrs={
        'standard_name': 'projection_x_coordinate',
        'axis': 'X',
        'units': 'm',
    },
    name='xc',
)

xc_bounds = xr.DataArray(
    data=[
        [500.0, 1250.0],
        [1250.0, 1750.0],
    ],
    dims=('xc', BOUNDS_DIM),
    coords={'xc': xc},
    name=f'xc_{BOUNDS_DIM}',
)

yc = xr.DataArray(
    data=[-4000.0, -3500.0, -3000.0],
    dims=('yc',),
    attrs={
        'standard_name': 'projection_y_coordinate',
        'axis': 'Y',
        'units': 'm',
    },
    name='yc',
)

yc_bounds = xr.DataArray(
    data=[
        [-4500.0, -3750.0],
        [-3750.0, -3000.0],
        [-3000.0, -2500.0],
    ],
    dims=('yc', BOUNDS_DIM),
    coords={'yc': yc},
    name=f'yc_{BOUNDS_DIM}',
)

lat_2d = xr.DataArray(
    data=[
        [89.5, 89.6],
        [99.5, 99.6],
        [109.5, 109.6],
    ],
    dims=('yc', 'xc'),
    name='lat',
    attrs={'standard_name': 'latitude', 'units': 'degrees_north'},
)

lon_2d = xr.DataArray(
    data=[
        [179.5, 179.6],
        [189.5, 189.6],
        [199.5, 199.6],
    ],
    dims=('yc', 'xc'),
    name='lon',
    attrs={'standard_name': 'longitude', 'units': 'degrees_east'},
)

temperature_2d = xr.DataArray(
    data=np.random.rand(4, 3, 2) * 300,
    dims=('time', 'yc', 'xc'),
    name='temperature',
    attrs={'standard_name': 'air_temperature', 'units': 'K'},
)


simple_2d = xr.Dataset(
    data_vars={'temperature': temperature_2d},
    coords={
        'time': time,
        'xc': xc,
        'yc': yc,
        'lat': lat_2d,
        'lon': lon_2d,
        'aux': aux,
    },
    attrs={
        'description': 'A synthetic CF-compliant dataset with two-dimensional coordinate variables.',
        'references': 'https://cfconventions.org/cf-conventions/cf-conventions.html#_two_dimensional_latitude_longitude_coordinate_variables',
    },
)

simple_2d_bounds = simple_2d.copy(deep=False)
simple_2d_bounds = simple_2d_bounds.assign_coords(
    {
        'time_bounds': time_bounds,
        'xc_bounds': xc_bounds,
        'yc_bounds': yc_bounds,
    }
).assign_attrs(
    {
        'description': simple_2d.attrs['description'].rstrip('.')
        + ' and boundary coordinate variables added.',
    }
)
simple_2d_bounds['time'].attrs['bounds'] = f'time_{BOUNDS_DIM}'
simple_2d_bounds['xc'].attrs['bounds'] = f'xc_{BOUNDS_DIM}'
simple_2d_bounds['yc'].attrs['bounds'] = f'yc_{BOUNDS_DIM}'
