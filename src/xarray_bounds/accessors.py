from collections.abc import Hashable, Iterator, Mapping

import cf_xarray  # noqa F401
import numpy.typing as npt
import xarray as xr
from xarray.core import formatting

from xarray_bounds.core import infer_bounds
from xarray_bounds.helpers import (
    mapping_or_kwargs,
    resolve_bounds_name,
    resolve_dim_name,
)
from xarray_bounds.types import IntervalClosed, IntervalLabel

__all__ = ['DataArrayBoundsAccessor', 'DatasetBoundsAccessor']


BOUNDS_ACCESSOR_NAME = 'bnds'


class BoundsAccessor[T: (xr.Dataset, xr.DataArray)](
    Mapping[str, xr.DataArray]
):
    """Accessor for Xarray boundary coordinates.

    This accessor returns a mapping of coordinate names to their boundary coordinates.
    """

    def __init__(self, obj: T) -> None:
        """Initialize a new ``Bounds`` object.

        A new bounds object is initialized every time a new xarray object is created.

        Parameters
        ----------
        obj : Dataset | DataArray
            The xarray object to add bounds to.
        """
        self._obj: T = obj.copy()

    @property
    def _data(self) -> Mapping[Hashable, xr.DataArray]:
        return {
            k: self._obj.coords[v[0]]
            for k, v in self._obj.cf.bounds.items()
            if k in self._obj.coords
        }

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        for k in self._data:
            yield str(k)

    def __len__(self):
        return len(self._data)

    @property
    def dims(self) -> set[Hashable]:
        """Set of dimension names for which bounds are defined."""
        return set(self._data.keys())

    @property
    def sizes(self) -> dict[Hashable, Mapping[Hashable, int]]:
        """Mapping of dimension names to their bounds sizes."""
        return {k: v.sizes for k, v in self._data.items()}

    def infer_bounds(
        self,
        *dims: str,
        closed: IntervalClosed | None = None,
        label: IntervalLabel | None = None,
    ) -> T:
        """Infer boundary coordinates for the specified dimensions and assign
        them to this object.

        Returns a new object with all the original original coordinates in
        addition to the new boundary coordinates.

        Parameters
        ----------
        *dims : str
            Names of dimension coordinates to add bounds for.
            If no ``dims`` are provided, bounds will be added for all
            available dimensions. The ``dims`` can be any values
            understood by :py:mod:`cf-xarray`.
        label : Literal['left', 'middle', 'right'], optional
            Which bin edge or midpoint the index labels.
        closed : Literal['left', 'right'], optional
            Which side of bin interval is closed.

        Returns
        -------
        Dataset
            A new object with bounds coordinates added.

        Raises
        ------
        KeyError
            If a key is not found in the object.
        ValueError
            If the coordinate is not 1D.
        TypeError
            If the coordinate does not have a compatible index.
        ValueError
            If a regular frequency cannot be inferred from the coordinate index.
        ValueError
            If bounds already exist for a specified dimension.
        """
        obj = self._obj.copy()
        keys = set(dims)
        if not keys:
            # default to all missing axes
            keys = set(self._obj.cf.axes) - set(self._obj.cf.bounds)
        coords = {}
        for key in keys:
            if key in getattr(obj, BOUNDS_ACCESSOR_NAME):
                raise ValueError(
                    f'Bounds already exist for dimension: {key!r}'
                )
            dim = resolve_dim_name(obj=obj, key=key)
            _bounds = infer_bounds(obj=obj[dim], closed=closed, label=label)
            coords.update({_bounds.name: _bounds})
        return obj.assign_coords(coords)

    def assign_bounds(
        self,
        bounds: Mapping[str, npt.ArrayLike] | None = None,
        **bounds_kwargs: npt.ArrayLike,
    ) -> T:
        """Assign new bounds coordinates to this object.

        Returns a new object with all the original data in addition to the
        new coordinates.

        Parameters
        ----------
        bounds : Mapping[str, ArrayLike], optional
            A mapping of dimension names or CF axis keys to bounds arrays.
        **bounds_kwargs : ArrayLike
            The keyword arguments form of ``bounds``.

        Returns
        -------
        T
            A new object with bounds coordinates added.

        Raises
        ------
        KeyError
            If the key is not found in the object.
        """
        obj = self._obj.copy()
        kwargs = mapping_or_kwargs(
            parg=bounds, kwargs=bounds_kwargs, func_name='assign_bounds'
        )
        for k, v in kwargs.items():
            dim = resolve_dim_name(obj=obj, key=str(k))
            name = resolve_bounds_name(dim)
            obj = obj.assign_coords({name: v})
            obj[dim].attrs['bounds'] = name
        return obj

    def __repr__(self) -> str:
        # TODO: replace with a custom representation
        return formatting._mapping_repr(
            self._data,
            title='Bounds',
            summarizer=formatting.summarize_variable,
            expand_option_name='display_expand_coords',
        )


@xr.register_dataset_accessor(BOUNDS_ACCESSOR_NAME)
class DatasetBoundsAccessor(BoundsAccessor):
    """An object for adding bounds coordinates to xarray Dataset objects."""

    pass


@xr.register_dataarray_accessor(BOUNDS_ACCESSOR_NAME)
class DataArrayBoundsAccessor:
    def __init__(self, obj: xr.DataArray) -> None:
        raise AttributeError(
            f'{obj.__class__.__name__!r} '
            f'object has no attribute {BOUNDS_ACCESSOR_NAME!r}.'
            'To manage bounds, you can convert the DataArray to a Dataset.'
        )
