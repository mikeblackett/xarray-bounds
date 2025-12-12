# FAQ

This page answers a few common questions about `xarray_bounds`.

## What are bounds and why should I care?

You can read the [CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html#_data_representative_of_cells) for a detailed explanation!

## Why not just use `cf_xarray`?

For many applications, {py:meth}`xarray.Dataset.cf.add_bounds` is a great choice. But this package might be useful if you require more extensive bounds-related features.

## The keys of `bnds` mapping are the coordinate names? Why?

The `bnds` mapping essentially creates a namespace for bounds, so you can just refer to them by the name of the variable they describe. We think this is intuitive. The name of the bounds variable itself is unimportant, what actually matters is the variable they describe.

## Why does the `bnds` accessor make a shallow copy of my `Dataset`?

The accessor wrapper is cheap to construct so it does a shallow copy
(`obj.copy(deep=False)`). Public accessor methods such as
`infer_bounds`, `assign_bounds`, and `drop_bounds` create and return new
copies and preserve the non-mutating API.

## Why isn't there a `bnds` accessor on `DataArray`?

In the current version of {py:mod}`xarray`, you can't add coordinates with new dimensions to {py:class}`xarray.DataArray`. So it is not currently possible to add CF-compliant bounds to `DataArray`s.  See [1475](https://github.com/pydata/xarray/issues/1475), [8005](https://github.com/pydata/xarray/issues/8005) and [10296](https://github.com/pydata/xarray/pull/10296) for more information. Hopefully [10296](https://github.com/pydata/xarray/pull/10296) will resolve this limitation.

## What about timezone / DST handling?

Some offsets with DST-aware timezones can be ambiguous. Functions that
infer intervals will raise a `ValueError` when an unsafe DST-aware offset
is detected; the recommended workaround is to convert timestamps to UTC
before inferring bounds.

## Do I have to use the accessor?

While the accessor is the main public interface, most of the core conversion and inference logic is available as standalone functions (in `src/xarray_bounds/core.py`).

## Where is the core logic implemented?

High-level APIs live in `src/xarray_bounds/accessors.py`. Core conversion
and inference logic lives in `src/xarray_bounds/core.py` and
`src/xarray_bounds/utilities.py`.

## How do I run tests locally?

Create a virtual environment and install the dev extras, then run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```
