# xarray-bounds

An xarray accessor and utilities for managing CF-compliant boundary (bounds)
coordinates. Provides a `bnds` accessor on `xarray.Dataset` and standalone
conversion utilities between bounds arrays and `pandas.Interval`s.

## Features

- **Organize** your `xarray.Dataset`'s bounds in a coords-like object
- **Infer**, **assign** and **drop** bounds with automatic CF **attribute management**
- Direct support for `cf_xarray` variable **aliasing**
- **Convert** `pandas.Index` to bounds `xarray.DataArray` and back

## Installation

`xarray-bounds` is still in development.

You can install it from source:

```bash
pip install git+https://github.com/mikeblackett/xarray-bounds
```

Or for development, using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mikeblackett/xarray-bounds
uv sync --dev
```

## Quick start

Example (infer bounds for a simple 1D coordinate):

```python
import xarray as xr

ds = xr.Dataset(coords={
  'y': ('y', [0, 1, 2])
})

# infer bounds for coordinate 'y' (labels at midpoint)
ds2 = ds.bnds.infer_bounds('y', label='middle')
print(ds2.bnds['y'])
```

This will add a coordinate named `y_bnds` with shape `(y, 2)` and set the
`bounds` attribute on the `y` coordinate.

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [docs](https://xarray-bounds.readthedocs.io/en/latest/).
