from collections.abc import Hashable, Iterator, Mapping
from typing import cast

import cf_xarray  # noqa F401
import numpy.typing as npt
import xarray as xr
from xarray.core import formatting

from xarray_bounds.core import infer_bounds
from xarray_bounds.helpers import (
    mapping_or_kwargs,
    resolve_variable_name,
)
from xarray_bounds.options import OPTIONS
from xarray_bounds.types import IntervalClosed, IntervalLabel

__all__ = ['DataArrayBoundsAccessor', 'DatasetBoundsAccessor']


BOUNDS_ACCESSOR_NAME = 'bnds'


class BoundsAccessor[T: (xr.Dataset, xr.DataArray)](
    Mapping[Hashable, xr.DataArray]
):
    """Xarray accessor for CF boundary variables.

    This accessor returns a mapping of variable names to boundary variables.
    """

    def __init__(self, obj: T) -> None:
        """Initialize a new ``Bounds`` object.

        This initializer makes a shallow copy of ``obj`` for performance
        reasons. Public accessor methods (`infer_bounds`, `assign_bounds`,
        `drop_bounds`) all explicitly create and return new copies and
        therefore preserve the non-mutating API.

        Parameters
        ----------
        obj : Dataset | DataArray
            The xarray object to add the bounds accessor to.
        """
        self._obj: T = obj.copy(deep=False)
        self._data = {
            key: self._obj.coords[value[0]]
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.coords
        }

    def _resolve_key(self, alias: Hashable) -> Hashable:
        """Resolve a canonical key in the bounds mapping.

        Parameters
        ----------
        alias : Hashable
            An alias for the key to resolve.
            Can be any value understood by :py:mod:`cf_xarray`.

        Returns
        -------
        Hashable
            The canonical key.
        """
        if alias in self._data:
            return alias
        key = resolve_variable_name(obj=self._obj, key=alias)
        if key in self._data:
            return key
        raise KeyError(f'No bounds found for alias: {alias!r}')

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        """Return the bounds for the specified key.

        Parameters
        ----------
        key : str
            The name of a coordinate.
            The name can be any alias understood by :py:mod:`cf_xarray`.

        Returns
        -------
        DataArray
            The bounds variable.

        Raises
        ------
        KeyError
            If the key is not found in the object.
        """
        canonical = self._resolve_key(key)
        return self._data[canonical]

    def __setitem__(self, key, value) -> None:
        raise TypeError(
            f'{self.__class__.__name__} object does not support item assignment. '
            "You can use the 'assign_bounds' method to add bounds."
        )

    def __delitem__(self, key) -> None:
        raise TypeError(
            f'{self.__class__.__name__} object does not support item deletion. '
            "You can use the 'drop_bounds' method to remove bounds."
        )

    def __contains__(self, key: object) -> bool:
        """Return True if bounds exist for the specified key.

        Parameters
        ----------
        key : Hashable
            The name of a coordinate.
            The name can be any alias understood by :py:mod:`cf_xarray`.

        Returns
        -------
        bool
            ``True`` if bounds exist for the specified key, ``False`` otherwise.
        """
        try:
            self._resolve_key(key)
            return True
        except Exception:
            return False

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the canonical keys in the bounds mapping.

        Returns
        -------
        Iterator[Hashable]
            An iterator over the canonical keys.
        """
        for key in self._data:
            yield str(key)

    def __len__(self) -> int:
        """Return the number of bounds in this object.

        Returns
        -------
        int
            The number of bounds in this object.
        """
        return len(self._data)

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Tuple of dimension names associated with this object's bounds."""
        dims = {dim for obj in self.values() for dim in obj.dims}
        # Return dims in the same order as _obj.dims
        return tuple(dim for dim in self._obj.dims if dim in dims)

    @property
    def variable_names(self) -> dict[Hashable, Hashable]:
        """Mapping of variable names to bounds names."""
        return {
            key: value
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.coords
        }

    @property
    def axes(self) -> dict[str, Hashable]:
        """Mapping of valid Axis standard names to bounds names."""
        return {
            key: value
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.cf.axes
        }

    @property
    def standard_names(self) -> dict[str, Hashable]:
        """Mapping of standard names to bounds names."""
        return {
            key: value
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.cf.standard_names
        }

    @property
    def coordinates(self) -> dict[str, Hashable]:
        """Mapping of valid Coordinate standard names to bounds names."""
        return {
            key: value
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.cf.coordinates
        }

    @property
    def sizes(self) -> dict[Hashable, Mapping[Hashable, int]]:
        """Mapping of dimension names to bounds sizes."""
        return {key: value.sizes for key, value in self._data.items()}

    def infer_bounds(
        self,
        *keys: str,
        closed: IntervalClosed | None = None,
        label: IntervalLabel | None = None,
    ) -> T:
        """Infer bounds for the specified coordinates and assign them to this object.

        Returns a new object with all the original data in addition to the new bounds.

        Parameters
        ----------
        *keys : str
            Names of coordinates.
            Can be any alias understood by :py:mod:`cf_xarray`.
            If no keys are provided, bounds will be inferred for all available axes.
        label : Literal['left', 'middle', 'right'], optional
            Which side or midpoint of the interval the index labels.
        closed : Literal['left', 'right'], optional
            Which side of the interval is closed.

        Returns
        -------
        T
            A new object with bounds variables added and appropriate attributes set.

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
        names = set(keys)
        if not names:
            names = set(self._obj.cf.axes) - set(self.axes)
        coords = {}
        for name in names:
            if name in self:
                raise ValueError(f'Bounds already exist for key: {name!r}')
            bounds = infer_bounds(obj=obj.cf[name], closed=closed, label=label)
            coords.update({bounds.name: bounds})
        return obj.assign_coords(coords)

    def assign_bounds(
        self,
        bounds: Mapping[str, npt.ArrayLike] | None = None,
        **bounds_kwargs: npt.ArrayLike,
    ) -> T:
        """Assign boundary variables to this object.

        Returns a new object with all the original data in addition to the
        new bounds.

        Parameters
        ----------
        bounds : Mapping[str, ArrayLike], optional
            A mapping of coordinate names to bounds arrays.
            The names can be any aliases understood by :py:mod:`cf_xarray`.
        **bounds_kwargs : ArrayLike
            The keyword arguments form of ``bounds``.

        Returns
        -------
        T
            A new object with bounds variables added.

        Raises
        ------
        KeyError
            If the key is not found in the object.
        """
        obj = self._obj.copy()
        kwargs = mapping_or_kwargs(
            parg=bounds, kwargs=bounds_kwargs, func_name='assign_bounds'
        )
        for key, value in kwargs.items():
            name = f'{resolve_variable_name(obj=obj, key=key)}_{OPTIONS["bounds_dim"]}'
            obj = obj.assign_coords({name: value})
            obj.cf[key].attrs['bounds'] = name
        return cast(T, obj)

    def drop_bounds(self, *keys: str) -> T:
        """Drop bounds variables for the specified coordinates.

        Returns a new object with the specified bounds variables and associated metadata removed.

        Parameters
        ----------
        *keys : str
            Names of coordinates to drop bounds for.
            The names can be any aliases understood by :py:mod:`cf_xarray`.

        Returns
        -------
        T
            A new object with bounds variables removed.

        Raises
        ------
        KeyError
            If a key is not found in the object.
        ValueError
            If bounds do not exist for a specified dimension.
        """
        obj = self._obj.copy()

        names = set(resolve_variable_name(obj, key) for key in keys)
        if not names:
            # default to all assigned bounds
            names = set(iter(self))

        for name in names:
            if 'bounds' in obj[name].attrs:
                del obj[name].attrs['bounds']
            if 'bounds' in obj.cf[name].encoding:
                del obj[name].encoding['bounds']
            obj = obj.drop_vars(self[name].name)  # type: ignore[arg-type]
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
class DatasetBoundsAccessor(BoundsAccessor[xr.Dataset]):
    """An object for adding bounds variables to xarray Dataset objects."""

    pass


@xr.register_dataarray_accessor(BOUNDS_ACCESSOR_NAME)
class DataArrayBoundsAccessor:
    def __init__(self, obj: xr.DataArray) -> None:
        raise AttributeError(
            f'{obj.__class__.__name__!r} '
            f'object has no attribute {BOUNDS_ACCESSOR_NAME!r}.'
            'To manage bounds, you can convert the DataArray to a Dataset.'
        )
