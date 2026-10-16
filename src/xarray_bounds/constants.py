from enum import StrEnum

__all__ = [
    "IntervalClosed",
    "IntervalLabel",
]


class IntervalLabel(StrEnum):
    """Options for interval labels"""

    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class IntervalClosed(StrEnum):
    """Options for interval closed"""

    LEFT = "left"
    RIGHT = "right"
