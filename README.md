# xarray-bounds

An xarray accessor and utilities for managing CF-compliant boundary (bounds)
coordinates. Provides a `bnds` accessor on `xarray.Dataset` and standalone
conversion utilities between bounds arrays and `pandas.Interval`s.

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

See the docs in `docs/source` (user guide, cookbook, and API).

