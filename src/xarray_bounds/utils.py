from collections.abc import Hashable, Mapping
from typing import Any, cast

import xarray as xr


def resolve_dim(obj: xr.Dataset | xr.DataArray, key: str) -> str:
    """Resolve a dimension name from an axis key.

    The key can be a dimension name or any key understood by ``cf-xarray``.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The xarray object to get the dimension name from
    key : str
        The dimension name or CF axis key

    Returns
    -------
    str
        The dimension name

    Raises
    ------
    KeyError
        If no dimension is found for the given axis.
    """
    if key in obj.dims:
        return key
    try:
        # There should only be one dimension for each axis...
        (dim,) = [v for v in obj.cf.axes[key] if v in obj.dims]
        if not dim:
            raise KeyError
    except KeyError as error:
        raise KeyError(f'No dimension found for key {key!r}. ') from error
    return dim


def mapping_or_kwargs[T](
    parg: Mapping[Any, T] | None,
    kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    """
    Return a mapping of arguments from either positional or keyword arguments.
    """
    if parg is None or parg == {}:
        return cast(Mapping[Hashable, T], kwargs)
    if kwargs:
        raise ValueError(
            f'cannot specify both keyword and positional arguments to {func_name}'
        )
    return parg
