import importlib.resources

import xarray as xr

import xarray_bounds.data as data_pkg

__all__ = ['load_dataset']


DATASETS = {
    'rainfall': 'rainfall_hadukgrid_uk_60km_day_20200101-20200131.nc',
    'tas_mon': 'tas_hadukgrid_uk_60km_mon_202001-202012.nc',
    'tas_seas': 'tas_hadukgrid_uk_60km_seas_202001-202012.nc',
    'tas_ann': 'tas_hadukgrid_uk_60km_ann_202001-202012.nc',
}

DATA_PATHS = {
    key: importlib.resources.files(data_pkg) / fname
    for key, fname in DATASETS.items()
}


def load_dataset(name: str) -> xr.Dataset:
    """Load a sample dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Available datasets are:
        - 'rainfall' - HadUK-Grid daily rainfall data
        - 'tas_mon' - HadUK-Grid monthly mean air temperature
        - 'tas_seas' - HadUK-Grid seasonal mean  air temperature
        - 'tas_ann' - HadUK-Grid annual mean air temperature

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    try:
        path = DATA_PATHS[name]
    except KeyError as e:
        raise ValueError(
            f'Dataset {name!r} not found. Available datasets: {list(DATASETS)}'
        ) from e

    with path.open('rb') as f:
        ds = xr.open_dataset(f, decode_coords='all')
        return ds.load()
