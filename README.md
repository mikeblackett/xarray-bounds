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

Quick install (recommended, editable install):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

See the docs in `docs/source` (user guide and API).
