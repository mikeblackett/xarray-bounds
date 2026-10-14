import pytest as pt

import xarray_bounds as xb
from xarray_bounds.options import OPTIONS

default_options = xb.get_options()


def test_unknown_option_raises():
    with pt.raises(ValueError):
        xb.set_options(unknown_option=True)


def test_bounds_dim():
    assert OPTIONS == default_options

    with pt.raises(ValueError):
        xb.set_options(bounds_dim=["unhashable", "mutable"])

    xb.set_options(bounds_dim="global")
    assert OPTIONS["bounds_dim"] == "global"

    with xb.set_options(bounds_dim="context"):
        assert OPTIONS["bounds_dim"] == "context"
    assert OPTIONS["bounds_dim"] == "global"


def test_interval_closed():
    assert OPTIONS == default_options

    with pt.raises(ValueError):
        xb.set_options(interval_closed="banana")

    xb.set_options(interval_closed="right")
    assert OPTIONS["interval_closed"] == "right"

    with xb.set_options(interval_closed="left"):
        assert OPTIONS["interval_closed"] == "left"
    assert OPTIONS["interval_closed"] == "right"


def test_label_side():
    assert OPTIONS == default_options

    with pt.raises(ValueError):
        xb.set_options(label_side="banana")

    xb.set_options(label_side="middle")
    assert OPTIONS["label_side"] == "middle"

    with xb.set_options(label_side="right"):
        assert OPTIONS["label_side"] == "right"
    assert OPTIONS["label_side"] == "middle"
