from collections.abc import Hashable, Iterator, Mapping
from typing import Generic, Literal, cast

import cf_xarray  # noqa F401
import xarray as xr
from xarray.core import formatting
from xarray.core.types import T_Xarray

from xarray_bounds._helpers import (
    mapping_or_kwargs,
    resolve_axis_name,
    resolve_variable_name,
)
from xarray_bounds.core import infer_bounds
from xarray_bounds.options import OPTIONS
from xarray_bounds.types import IntervalClosed, IntervalLabel

__all__ = ["DataArrayBoundsAccessor", "DatasetBoundsAccessor"]


BOUNDS_ACCESSOR_NAME = "bnds"


class BoundsAccessor(Mapping[Hashable, xr.DataArray], Generic[T_Xarray]):
    """Xarray accessor for CF boundary variables.

    This accessor returns a mapping of variable names to boundary variables.
    """

    def __init__(self, obj: T_Xarray) -> None:
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
        self._obj: T_Xarray = obj.copy(deep=False)
        self._data = {
            key: self._obj.coords[value[0]]
            for key, value in self._obj.cf.bounds.items()
            if key in self._obj.coords
        }

    def _resolve_key(self, alias: Hashable) -> Hashable:
        """Resolve an alias to a canonical key in the bounds mapping.

        Parameters
        ----------
        alias : Hashable
            An alias for a canonical key.
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
        raise KeyError(f"No bounds found for alias: {alias!r}")

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        """Return the bounds for the specified key.

        Parameters
        ----------
        key : str
            The name of a coordinate.
            Can be any alias understood by :py:mod:`cf_xarray`.

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
            f"{self.__class__.__name__} object does not support item assignment. "
            "You can use the 'assign_bounds' method to add bounds."
        )

    def __delitem__(self, key) -> None:
        raise TypeError(
            f"{self.__class__.__name__} object does not support item deletion. "
            "You can use the 'drop_bounds' method to remove bounds."
        )

    def __contains__(self, key: object) -> bool:
        """Return True if bounds exist for the specified key.

        Parameters
        ----------
        key : Hashable
            The name of a coordinate.
            Can be any alias understood by :py:mod:`cf_xarray`.

        Returns
        -------
        bool
            ``True`` if bounds exist for the specified key, ``False`` otherwise.
        """
        try:
            self._resolve_key(key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the canonical keys in the bounds mapping.

        Returns
        -------
        Iterator[Hashable]
            An iterator over the canonical keys.
        """
        yield from self._data

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
    def variable_names(self) -> dict[Hashable, tuple[str, ...]]:
        """Mapping of variable names to bounds names."""
        return {
            key: self._obj.cf.bounds[key]
            for key in self._obj.variables
            if key in self._obj.cf.bounds
        }

    @property
    def axes(self) -> dict[str, tuple[str, ...]]:
        """Mapping of CF Axis names to bounds names."""
        return self._bounds_by_cf_key("axes")

    @property
    def standard_names(self) -> dict[str, tuple[str, ...]]:
        """Mapping of CF standard names to bounds names."""
        return self._bounds_by_cf_key("standard_names")

    @property
    def coordinates(self) -> dict[str, tuple[str, ...]]:
        """Mapping of CF Coordinate names to bounds names."""
        return self._bounds_by_cf_key("coordinates")

    @property
    def sizes(self) -> dict[Hashable, Mapping[Hashable, int]]:
        """Mapping of dimension names to bounds sizes."""
        return {key: value.sizes for key, value in self._data.items()}

    def _bounds_by_cf_key(
        self,
        cf_key: Literal["standard_names", "axes", "coordinates"],
    ) -> dict[str, tuple[str, ...]]:
        """Return a mapping of CF names to bounds.

        The order is the same as ``cf_xarray``."""
        return {
            key: self._obj.cf.bounds[key]
            for key in getattr(self._obj.cf, cf_key)
            if key in self._obj.cf.bounds
        }

    def infer_bounds(
        self,
        *keys: str,
        closed: IntervalClosed | None = None,
        label: IntervalLabel | None = None,
    ) -> T_Xarray:
        """Infer bounds for the specified variables and assign them to this object.

        Returns a new object with all the original data in addition to the new bounds.

        Parameters
        ----------
        *keys : str
            Names of coordiantes to infer bounds for.
            Can be any alias understood by :py:mod:`cf_xarray`.
            If no keys are provided, bounds will be inferred for all available CF axes.
        label : Literal['left', 'middle', 'right'], optional
            Which side or midpoint of the interval the index labels.
        closed : Literal['left', 'right'], optional
            Which side of the interval is closed.

        Returns
        -------
        T_Xarray
            A new object with bounds variables added and appropriate attributes set.

        Raises
        ------
        KeyError
            If a key is not found in the object.
        ValueError
            If no keys are specified and no CF axes are found.
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

        axes = {resolve_axis_name(obj, key) for key in keys}
        if not axes:
            # all CF axes that don't already have bounds
            axes = set(obj.cf.axes) - set(self.axes)
        if not axes:
            raise ValueError(
                "no CF axes found to infer bounds for."
                " You can try `obj.cf.guess_coord_axis()` to discover CF axis variables."
            )

        coords = {}
        # add bounds in the same order as thier axes
        ordered = [a for a in obj.cf.axes if a in axes]
        for axis in ordered:
            if axis in self.axes:
                raise ValueError(f"bounds already exist for axis: {axis!r}")
            bounds = infer_bounds(obj=obj.cf[axis], closed=closed, label=label)
            coords.update({bounds.name: bounds})
        return obj.assign_coords(coords)

    def assign_bounds(
        self,
        bounds: Mapping[str, xr.DataArray] | None = None,
        **bounds_kwargs: xr.DataArray,
    ) -> T_Xarray:
        """Assign boundary variables to this object.

        Returns a new object with all the original data in addition to the
        new bounds.

        Parameters
        ----------
        bounds : Mapping[str, DataArray], optional
            A mapping of coordinate names to bounds arrays.
            The keys can be any alias understood by :py:mod:`cf_xarray`.
        **bounds_kwargs : DataArray
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
            parg=bounds, kwargs=bounds_kwargs, func_name="assign_bounds"
        )
        for key, value in kwargs.items():
            var = resolve_variable_name(obj=obj, key=key)
            name = f"{var}_{OPTIONS['bounds_dim']}"
            obj = obj.assign_coords({name: value})
            obj[var].attrs["bounds"] = name
        return cast(T_Xarray, obj)

    def drop_bounds(self, *keys: str) -> T_Xarray:
        """Drop bounds variables for the specified coordinates.

        Returns a new object with the specified bounds variables and associated metadata removed.

        Parameters
        ----------
        *keys : str
            Names of variables whose bounds to drop.
            The names can be any alias understood by :py:mod:`cf_xarray`.

        Returns
        -------
        T
            A new object with bounds variables removed.

        Raises
        ------
        KeyError
            If a variable does not exist for a specified key.
        KeyError
            If bounds do not exist for a specified key.
        """
        obj = self._obj.copy()

        if not keys:
            # default to all assigned bounds
            variables = set(self)
        else:
            variables = {resolve_variable_name(obj, key) for key in keys}

        bounds: set[str] = set()
        for variable in variables:
            bounds |= set(self.variable_names[variable])
            if "bounds" in obj[variable].attrs:
                del obj[variable].attrs["bounds"]
            if "bounds" in obj.cf[variable].encoding:
                del obj[variable].encoding["bounds"]
        obj = obj.drop_vars(bounds)
        return obj

    def __repr__(self) -> str:
        # TODO: replace with a custom representation
        return formatting._mapping_repr(
            self._data,
            title="Bounds",
            summarizer=formatting.summarize_variable,
            expand_option_name="display_expand_coords",
        )


@xr.register_dataset_accessor(BOUNDS_ACCESSOR_NAME)
class DatasetBoundsAccessor(BoundsAccessor[xr.Dataset]):
    """An object for adding bounds variables to xarray Dataset objects."""


@xr.register_dataarray_accessor(BOUNDS_ACCESSOR_NAME)
class DataArrayBoundsAccessor:
    def __init__(self, obj: xr.DataArray) -> None:
        raise NotImplementedError(
            f"the {BOUNDS_ACCESSOR_NAME!r} accessor is not available for DataArray objects."
            " To manage bounds, convert the DataArray to a Dataset (e.g. `obj.to_dataset()`)."
        )
