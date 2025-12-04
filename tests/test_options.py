import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
import xarray as xr

import xarray_bounds
from xarray_bounds.options import OPTIONS


@hp.given(value=st.text(min_size=1, max_size=10))
def test_set_options(value: str):
    """Test the set_options function."""
    xarray_bounds.set_options(bounds_dim=value)
    assert OPTIONS['bounds_dim'] == value


@hp.given(value=st.text(min_size=1, max_size=10))
def test_set_options_context_manager(value: str):
    """Test the set_options function."""
    with xarray_bounds.set_options(bounds_dim=value):
        assert OPTIONS['bounds_dim'] == value


@hp.given(value=st.text(min_size=1, max_size=10))
def test_set_bounds_dim(value: str):
    """Test that set_options works as a context manager."""
    with xarray_bounds.set_options(bounds_dim=value):
        ds = xr.tutorial.open_dataset('air_temperature').bnds.infer_bounds(
            'time'
        )
        assert value in ds.bnds['time'].dims
        assert ds.time.attrs['bounds'] == f'time_{value}'


@hp.given(
    key=st.text(),
    value=st.one_of(st.integers(), st.floats(), st.text(), st.none()),
)
def test_invalid_option_raises(key: str, value):
    """Setting an invalid option should raise a ValueError."""
    hp.assume(key not in OPTIONS)
    with pt.raises(ValueError):
        xarray_bounds.set_options(**{key: value})
