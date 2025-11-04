from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, TypeIs

import pandas as pd

__all__ = [
    'AxisKey',
    'ClosedSide',
    'IntervalClosed',
    'IntervalLabel',
    'LabelSide',
    'is_datetime_index',
    'is_date_offset',
    'is_interval_index',
    'is_multi_index',
    'resolve_interval_closed',
    'resolve_interval_label',
]

type IntervalLabel = Literal['left', 'middle', 'right']

type IntervalClosed = Literal['left', 'right']

type AxisKey = Literal['T', 'Z', 'Y', 'X']

type AxisToBounds = Mapping[CFAxis, Sequence[str]]


class CFAxis(StrEnum):
    """Options for axis keys"""

    T = 'T'
    Z = 'Z'
    Y = 'Y'
    X = 'X'


class LabelSide(StrEnum):
    """Options for interval labels"""

    LEFT = 'left'
    MIDDLE = 'middle'
    RIGHT = 'right'


class ClosedSide(StrEnum):
    """Options for interval closed"""

    LEFT = 'left'
    RIGHT = 'right'


def resolve_interval_label(label: IntervalLabel) -> LabelSide:
    return LabelSide(label)


def resolve_interval_closed(label: IntervalClosed) -> ClosedSide:
    return ClosedSide(label)


def is_datetime_index(index: object) -> TypeIs[pd.DatetimeIndex]:
    """Check if the object is a datetime index."""
    return isinstance(index, pd.DatetimeIndex)


def is_interval_index(index: object) -> TypeIs[pd.IntervalIndex]:
    """Check if the object is an interval index."""
    return isinstance(index, pd.IntervalIndex)


def is_multi_index(index: object) -> TypeIs[pd.MultiIndex]:
    """Check if the object is a MultiIndex."""
    return isinstance(index, pd.MultiIndex)


def is_date_offset(offset: object) -> TypeIs[pd.DateOffset]:
    """Check if the object is a DateOffset."""
    return isinstance(offset, pd.DateOffset)
