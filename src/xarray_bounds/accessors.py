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

__all__ = ['DataArrayBounds', 'DatasetBounds']

CF_AXES: set[AxisKey] = {'T', 'X', 'Y', 'Z'}


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
        self._obj = obj.copy()

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
            A new dataset with bounds coordinates added.

        Raises
        ------
        KeyError
            If a key is not found in the dataset.
        """
        dataset = self._obj.copy()
        keys = set(key_args)
        if not keys:
            # default to all missing axes
            keys = set(self._obj.cf.axes) - set(self.axes)
        # normalize keys to variable names
        dims = {resolve_dim_name(obj=dataset, key=key) for key in keys}
        coords = {}
        for dim in dims:
            _bounds = infer_bounds(
                obj=dataset[dim], closed=closed, label=label
            )
            coords.update({_bounds.name: _bounds})
        return dataset.assign_coords(coords)

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
            if dim not in self._obj.dims:
                raise ValueError(
                    f'Dimension {dim!r} does exist in object {self._obj!r}.'
                )
            bounds_name = f'{dim}_bounds'
            coords.update({bounds_name: data})
            obj = obj.assign_coords(coords)
            obj[dim].attrs['bounds'] = bounds_name
        return obj

    def to_index(self, key: str) -> pd.IntervalIndex:
        """Return the specified bounds as a ``pandas.IntervalIndex``.

        The closed side of the interval is determined by the `closed` attribute
        of the bounds coordinate. If the closed side is not specified, it defaults
        to 'left'. If the closed side attribute is not a valid
        ``IntervalClosed``, a ``ValueError`` is raised.

        Parameters
        ----------
        key : str
            The name or CF axis key of the bounds coordinate to convert to a
            ``pandas.IntervalIndex``.
        Returns
        -------
        pd.IntervalIndex
            The bounds as a ``pandas.IntervalIndex``.
        """
        bounds = self[key]
        attr = bounds.attrs.get('closed', 'left')
        try:
            closed = ClosedSide(attr)
        except ValueError:
            raise ValueError(
                f"Invalid closed attr: {attr}. Must be 'left' or 'right'; got {attr}"
            )
        return pd.IntervalIndex.from_arrays(
            *bounds.values.transpose(), closed=closed.value
        )

    def to_midpoint(self, key: str) -> xr.DataArray:
        """Return the specified bounds with the time labels as the midpoints of the bounds.

        The midpoints are calculated by converting the bounds to a pandas
        `IntervalIndex`, and then taking the `IntervalIndex.mid` attribute. If
        the bounds are datetime-like, the midpoints are normalized to midnight.
        """
        interval = self.to_index(key)
        midpoint = interval.mid
        if isinstance(midpoint, pd.DatetimeIndex):
            midpoint = midpoint.normalize()  # pyright: ignore[reportAttributeAccessIssue]
        dim = resolve_dim_name(obj=self._obj, key=key)
        return xr.DataArray(
            data=midpoint,
            dims=(dim,),
            name=dim,
        )


@xr.register_dataset_accessor('bounds')
class DatasetBounds(Bounds):
    pass


@xr.register_dataarray_accessor('bounds')
class DataArrayBounds:
    def __init__(self, *args, **kwargs) -> None:
        raise NotYetImplementedError(
            'bounds are not currently supported for DataArrays.'
            'To add bounds, you can convert the DataArray to a Dataset.'
        )
