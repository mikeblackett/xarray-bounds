# {{project}} documentation

**Version:** {{version}}

An {py:mod}`xarray` accessor for managing
[CF](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#cell-boundaries)
boundary variables.

:::{important}
This project is under active development. Frequent and breaking changes are expected.

Significant breaking changes are anticipated with {py:mod}`xarray`'s upcoming
support for CF interval indexes. See
<https://github.com/pydata/xarray/pull/10296> for more information.
:::

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

user_guide
api
:::

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Next steps](#next-steps)

## Overview

This package adds the `bnds` accessor to {py:class}`xarray.Dataset`. The `bnds` accessor recognizes bounds variables, collects them in a {py:attr}`~xarray.Dataset.coords`-like mapping and provides some useful methods. You can infer, assign and drop bounds -- all with automatic CF attribute management. Check out the [User Guide](user_guide) for a general introduction.

## Features

- **Organize** your {py:class}`~xarray.Dataset`'s bounds in a coords-like object
- **Infer**, **assign** and **drop** bounds with automatic CF **attribute management**
- Direct support for {py:mod}`cf_xarray` variable **aliasing**
- **Convert** {py:class}`pandas.Index` to bounds {py:class}`~xarray.DataArray` and back

## Installation

Install locally using an editable install (recommended for development):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Alternatively, for a minimal install:

```bash
pip install -e .
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

Check out the [User guide](user_guide.ipynb).
