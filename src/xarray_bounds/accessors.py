from collections.abc import Iterator, Mapping
from typing import Literal

import cf_xarray  # noqa F401
import numpy.typing as npt
import pandas as pd
import xarray as xr

from xarray_bounds.exceptions import NotYetImplementedError
from xarray_bounds.helpers import infer_bounds
from xarray_bounds.types import AxisKey, ClosedSide
from xarray_bounds.utilities import resolve_dim_name, mapping_or_kwargs
from xarray_bounds.options import OPTIONS

__all__ = ['DataArrayBounds', 'DatasetBounds']

CF_AXES: set[AxisKey] = {'T', 'X', 'Y', 'Z'}
DATA_ARRAY_BNDS_ACCESSOR_NAME = 'bnds'


class Bounds[T: (xr.Dataset, xr.DataArray)](Mapping[str, xr.DataArray]):
    """An object for adding bounds coordinates to xarray objects."""

    def __init__(self, obj: T) -> None:
        """Initialize a new ``Bounds`` object.

        A new bounds object is initialized every time a new xrarray object is created.

        Parameters
        ----------
        obj : Dataset
            The xarray object to add bounds to.
        """
        self._obj: T = obj.copy()

    def __getitem__(self, dim: str) -> xr.DataArray:
        """Return the bounds for the specified dimension.

        The bounds can be referenced by their coordinate name, CF standard
        name, or by their corresponding CF axis key.

        Parameters
        ----------
        dim : str
            The name or CF axis key of the dimension coordinate to get bounds for.

        Returns
        -------
        DataArray
            The bounds data array.

        Raises
        ------
        KeyError
            If the bounds are not found for the specified dimension coordinate.
        """
        try:
            # TODO: What about multiple bounds?
            key = self._obj.cf.bounds[dim][0]
        except KeyError as error:
            raise KeyError(f'No bounds found for axis {dim!r}.') from error
        return self._obj.coords[key]

    def __iter__(self) -> Iterator[str]:
        for k in self._obj.cf.bounds:
            yield str(k)

    def __len__(self):
        return len(self._obj.cf.bounds)

    @property
    def axes(self) -> dict[str, str]:
        """Mapping of CF axes to bounds-coordinate names."""
        return {k: v for k, v in self._obj.cf.bounds.items() if k in CF_AXES}

    @property
    def dims(self) -> set[str]:
        """Set of dimension names for which bounds are defined."""
        return {k for k in self._obj.cf.bounds if k in self._obj.dims}

    def infer_bounds(
        self,
        key: str,
        closed: Literal['left', 'right'] | None = None,
        label: Literal['left', 'middle', 'right'] | None = None,
    ) -> xr.DataArray:
        """Infer bounds for the specified dimension.

        Parameters
        ----------
        key : str
            The name or CF axis key of the dimension coordinate to infer bounds for.
        closed : {'left', 'right'}, optional
            Which side of the interval bin is closed.
        label : {'left', 'middle', 'right'}, optional
            Which bin edge or midpoint the index labels.

        Returns
        -------
        xarray.DataArray
            A DataArray containing the inferred bounds.

        Raises
        ------
        KeyError
            If the key is not found in the object.
        """
        obj = self._obj.copy()
        dim = resolve_dim_name(obj=obj, key=key)
        return infer_bounds(obj=obj[dim], closed=closed, label=label)

    def add_bounds(
        self,
        *key_args: str,
        closed: Literal['left', 'right'] | None = None,
        label: Literal['left', 'middle', 'right'] | None = None,
    ) -> T:
        """Add bounds coordinates for the specified dimensions.

        Returns a new object with all the original data in addition to the new
        coordinates.

        The `closed` and `label` parameters will apply to all bounds. If you
        want to add bounds with different parameters, you will need to call
        this method multiple times.

        Parameters
        ----------
        *key_args : str
            The names or CF axis keys of the dimension coordinates to add
            bounds for. If no keys are provided, bounds will be added for all
            applicable dimensions.
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
        """
        obj = self._obj.copy()
        keys = set(key_args)
        if not keys:
            # default to all missing axes
            keys = set(self._obj.cf.axes) - set(self.axes)
        coords = {}
        for key in keys:
            if key in obj.bounds:
                raise KeyError(f'Bounds already exist for dimension: {key!r}')
            _bounds = self.infer_bounds(key=key, closed=closed, label=label)
            coords.update({_bounds.name: _bounds})
        return obj.assign_coords(coords)

    def assign_bounds(
        self,
        bounds: Mapping[str, npt.ArrayLike] | None = None,
        **bounds_kwargs: npt.ArrayLike,
    ) -> T:
        """Assign new bounds coordinates to this object.

        Returns a new object with all the original data in addition to the new coordinates.
        """
        obj = self._obj.copy()
        kwargs = mapping_or_kwargs(
            parg=bounds, kwargs=bounds_kwargs, func_name='assign_bounds'
        )
        coords = {}
        for dim, data in kwargs.items():
            dim = resolve_dim_name(obj=obj, key=str(dim))
            bounds_name = f'{dim}_f{OPTIONS["bounds_name"]}'
            coords.update({bounds_name: data})
            obj = obj.assign_coords(coords)
            obj[dim].attrs['bounds'] = bounds_name
        return obj


@xr.register_dataset_accessor('bounds')
class DatasetBounds(Bounds):
    """An object for adding bounds coordinates to xarray Dataset objects."""

    pass


@xr.register_dataarray_accessor('bounds')
class DataArrayBounds:
    def __init__(self, *args, **kwargs) -> None:
        raise NotYetImplementedError(
            'bounds are not currently supported for DataArrays.'
            'To add bounds, you can convert the DataArray to a Dataset.'
        )


@xr.register_dataarray_accessor(DATA_ARRAY_BNDS_ACCESSOR_NAME)
class DataArrayBnd:
    def __init__(self, obj: xr.DataArray) -> None:
        if obj.ndim != 2:
            raise ValueError(
                f'.{DATA_ARRAY_BNDS_ACCESSOR_NAME!r} '
                'accessor only works for 2D DataArrays.'
            )
        if obj.dims[1] != OPTIONS['bounds_name']:
            raise ValueError(
                f'.{DATA_ARRAY_BNDS_ACCESSOR_NAME!r} '
                f'accessor only works for 2D DataArrays with a {OPTIONS["bounds_name"]!r} dimension.'
            )
        self._obj = obj.copy()

    def to_interval(self) -> pd.IntervalIndex:
        """Return the bounds as a ``pandas.IntervalIndex``."""
        attr = self._obj.attrs.get('closed', ClosedSide.DEFAULT)
        closed = ClosedSide(attr)
        return pd.IntervalIndex.from_arrays(
            *self._obj.values.transpose(), closed=closed.value
        )

    @property
    def length(self) -> xr.DataArray:
        """Lengths of the bounds intervals."""
        dim = self._obj.dims[0]
        return xr.DataArray(
            data=self.to_interval().length,
            dims=(dim,),
            coords={dim: self._obj.cf[dim]},
            name=f'{dim}_length',
        )

    @property
    def midpoint(self) -> xr.DataArray:
        """Midpoints of the bounds intervals."""
        midpoint = self.to_interval().mid
        if isinstance(midpoint, pd.DatetimeIndex):
            midpoint = midpoint.normalize()  # pyright: ignore[reportAttributeAccessIssue]
        dim = self._obj.dims[0]
        return xr.DataArray(
            data=midpoint,
            dims=(dim,),
            name=dim,
        )
