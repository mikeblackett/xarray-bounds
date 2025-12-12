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
examples
faq
api
:::

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Next steps](#next-steps)

## Overview

This package adds a `bnds` accessor to {py:class}`xarray.Dataset`. The
`bnds` accessor recognizes bounds variables and collects them in a
{py:class}`xarray.Coordinates`-like mapping. The mapping provides methods to
infer, assign and drop bounds (see the [User Guide](user_guide)), with automatic CF
attribute management. You can refer to variables by name or any alias supported
by {py:mod}`cf_xarray`.

## Features

- **Organize**: access all your {py:class}`~xarray.Dataset`'s bounds in a dict-like object
- **Infer**: automagically infer bounds from indexed coordinates[^1]
- **Attribute management**: manipulate bounds with automatic CF attribute
  management
- **CF alias support**: refer to variables using any alias understood by {py:mod}`cf_xarray`
- **Bounds round-tripping**: convert {py:class}`pandas.Index` to bounds
  `~xarray.DataArray` and back
- **Flexible API**: manage bounds through the accessor interface or utilize the
  core logic as standalone functions

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
print(ds2.coords['y_bnds'])
```

This will add a coordinate named `y_bnds` with shape `(y, 2)` and set the
`bounds` attribute on the `y` coordinate.

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [User guide](user_guide.ipynb) for an in depth introduction and
the [Cookbook](cookbook.ipynb) for examples.

See the [FAQ](FAQ.md) for common questions about accessor semantics and
timezone handling.

[^1]: To infer bounds the index must be monotonic and regularly-spaced.