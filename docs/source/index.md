---
kernelspec:
  name: xarray-bounds
---

# {{project}} documentation

**Version:** {{version}}

An {py:mod}`xarray` accessor for managing
[CF](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#cell-boundaries)
boundary variables.

:::{important}
This project is under active development. Frequent and breaking changes are expected.

Significant breaking changes are anticipated with {py:mod}`xarray`'s upcomming
support for CF interval indexes.

See <https://github.com/pydata/xarray/pull/10296> for more information.
:::

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

user_guide
cookbook
api
:::

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)

## Overview

Gridded datasets have dimension coordinates that represent points on a grid.
These points lie somewhere within or upon the boundaries of grid cells. If the
vertices of the cell boundaries are not described, the meaning of the coordinates is ambiguous.

The CF convention recommends adding "boundary variables" to delimit cell
boundaries. A boundary variable is a coordinate that has one more
dimension--the vertex dimension--than its associated coordinate. So a
coordinate is a variable that describes a dimension and a boundary variable
is a variable that describes a coordinate.
  
This extra dimension can be modelled with xarray's {py:class}`~xarray.Dataset`.
But because the vertex dimension is not a dimension of the data variable(s),
many routine operations like reduction, groupby, regridding etc drop the
boundary variables. This means that boundary variables must be carefully maintained.
  
## Features

- **Bounds mapping**: access bounds by  
- **Assign**: assign boundary variables with 
- **Infer bounds**: automagically infer boundary variables from
  indexed[^1] coordinates
- **Attribute management**: perform boundary operations with automatic
  CF attribute management

## Installation

You can install a local pre-release copy of `xarray_bounds` by cloning this
repository:

```bash
mkdir xarray-bounds && cd xarray-bounds
git clone 'https://github.com/mikeblackett/xarray-bounds'
uv pip install -e . 
```

## Quick start

```python
import xarray as xr

import xarray_bounds as xrb  # noqa F401

y = xr.DataArray(
    data=range(3),
    dims='y',
)
ds = xr.Dataset(coords={'y': range(3)})
ds
```

```text
<xarray.DataArray 'y' (y: 3)> Size: 24B
array([0, 1, 2])
Coordinates:
  * y        (y) int64 24B 0 1 2
Attributes:
    bounds:   y_bnds
```

```python
ds2 = ds.bnds.infer_bounds('y', label='middle')
print(ds2.coords['y'], ds2.bnds['y'], sep='\n\n')
```

```text
<xarray.DataArray 'y_bnds' (y: 3, bnds: 2)> Size: 48B
array([[-0.5,  0.5],
       [ 0.5,  1.5],
       [ 1.5,  2.5]])
Coordinates:
  * y        (y) int64 24B 0 1 2
    y_bnds   (y, bnds) float64 48B -0.5 0.5 0.5 1.5 1.5 2.5
Dimensions without coordinates: bnds
Attributes:
    closed:   left
```

## Next steps

Check out the [User guide](user_guide.ipynb) for an in depth introduction and
the [Cookbook](cookbook.ipynb) for examples.

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

[^1]: To infer bounds the index must be monotonic and regularly-spaced.
