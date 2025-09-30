import pytest
import xarray as xr

import hypothesis.strategies as st
import hypothesis as hp
from xarray_bounds.utilities import mapping_or_kwargs, resolve_dim_name


@hp.given(
    key=st.sampled_from(
        [
            # cf axis
            'Y',
            # name
            'lat',
            # standard_name
            'latitude',
        ]
    )
)
def test_resolve_dim(key: str):
    """Should return the dimension name if it exists."""
    ds = xr.tutorial.load_dataset('air_temperature')
    assert resolve_dim_name(ds, key) == 'lat'


@hp.given(key=st.text())
def test_resolve_dim_raises_key_error(key: str):
    """Should raise a KeyError if the dimension cannot be resolved."""
    da = xr.DataArray()
    with pytest.raises(KeyError):
        resolve_dim_name(da, key)


@pytest.mark.parametrize(
    'pargs, kwargs, expected',
    [
        (None, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}),
        ({'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}),
        ({}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}),
    ],
)
def test_mapping_or_kwargs(pargs, kwargs, expected):
    actual = mapping_or_kwargs(
        parg=pargs, kwargs=kwargs, func_name='test_func'
    )
    assert actual == expected


@pytest.mark.parametrize(
    'pargs, kwargs, func_name',
    [
        ({'a': 1}, {'b': 2}, 'test_func'),
    ],
)
def test_mapping_or_kwargs_raises_value_error(pargs, kwargs, func_name):
    with pytest.raises(ValueError):
        mapping_or_kwargs(pargs, kwargs, func_name)
