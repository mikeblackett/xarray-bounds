from enum import StrEnum
from typing import Literal, TypeAlias, TypeIs

import pandas as pd

__all__ = [
    'ClosedSide',
    'IntervalClosed',
    'IntervalLabel',
    'LabelSide',
    'is_datetime_index',
    'is_date_offset',
    'is_interval_index',
    'is_multi_index',
]

IntervalLabel: TypeAlias = Literal['left', 'middle', 'right']
"""Options for interval labels. Values are members of the LabelSide enum."""

IntervalClosed: TypeAlias = Literal['left', 'right']
"""Options for interval closed. Values are members of the ClosedSide enum."""


class LabelSide(StrEnum):
    """Options for interval labels"""

    LEFT = 'left'
    MIDDLE = 'middle'
    RIGHT = 'right'


class ClosedSide(StrEnum):
    """Options for interval closed"""

    LEFT = 'left'
    RIGHT = 'right'


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
