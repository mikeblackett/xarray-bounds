# API reference

## Top-level API

```{eval-rst}
..  autosummary::
    :toctree: generated/

    ~xarray_bounds.bounds_to_interval
    ~xarray_bounds.infer_bounds
    ~xarray_bounds.interval_to_bounds
    ~xarray_bounds.set_options
```

## Dataset

### Attributes

```{eval-rst}
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_attribute.rst

    ~xarray.Dataset.bnds.dims
    ~xarray.Dataset.bnds.variable_names
    ~xarray.Dataset.bnds.axes
    ~xarray.Dataset.bnds.standard_names
    ~xarray.Dataset.bnds.coordinates
    ~xarray.Dataset.bnds.sizes
```

### Methods

```{eval-rst}
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    ~xarray.Dataset.bnds.infer_bounds
    ~xarray.Dataset.bnds.assign_bounds
    ~xarray.Dataset.bnds.drop_bounds
```

### Dictionary interface

```{eval-rst}
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    ~xarray.Dataset.bnds.__getitem__
    ~xarray.Dataset.bnds.__contains__
    ~xarray.Dataset.bnds.__iter__
    ~xarray.Dataset.bnds.__len__
    ~xarray.Dataset.bnds.get
    ~xarray.Dataset.bnds.keys
    ~xarray.Dataset.bnds.items
    ~xarray.Dataset.bnds.values
```

## DataArray

The `bnds` accessor is not currently available for {py:class}`~xarray.DataArray` objects because it depends on developmental features of {py:mod}`xarray`. See <https://github.com/pydata/xarray/issues/1475> for more information.

:::{important}
The `bnds` accessor is not currently available for {py:class}`~xarray.DataArray` objects. See <https://github.com/pydata/xarray/issues/1475> for more information.
:::
