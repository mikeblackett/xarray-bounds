from collections.abc import Mapping, Sequence
from typing import Literal, TypeIs
from enum import StrEnum

import pandas as pd

__all__ = [
    'AxisKey',
    'ClosedSide',
    'IntervalClosed',
    'IntervalLabel',
    'LabelSide',
    'is_datetime_index',
    'resolve_interval_closed',
    'resolve_interval_label',
]

type IntervalLabel = Literal['left', 'middle', 'right']

type IntervalClosed = Literal['left', 'right', 'both', 'neither']

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
    BOTH = 'both'
    NEITHER = 'neither'


def resolve_interval_label(label: IntervalLabel) -> LabelSide:
    return LabelSide(label)


def resolve_interval_closed(label: IntervalClosed) -> ClosedSide:
    return ClosedSide(label)


def is_datetime_index(index: object) -> TypeIs[pd.DatetimeIndex]:
    """Check if the index is a datetime index."""
    return isinstance(index, pd.DatetimeIndex)
