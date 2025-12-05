..  currentmodule:: xarray_bounds

..  _api:

#############
API reference
#############

*******************
Top-level functions
*******************


..  autosummary::
    :toctree: generated/
    :template: autosummary/module_function.rst

    bounds_to_interval
    infer_bounds
    interval_to_bounds
    set_options

.. currentmodule:: xarray


*********
Accessor
*********

Dataset
=======

Attributes
~~~~~~~~~~
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_attribute.rst

    Dataset.bnds.dims
    Dataset.bnds.variable_names
    Dataset.bnds.axes
    Dataset.bnds.standard_names
    Dataset.bnds.coordinates
    Dataset.bnds.sizes

Methods
~~~~~~~
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    Dataset.bnds.infer_bounds
    Dataset.bnds.assign_bounds
    Dataset.bnds.drop_bounds

Dictionary interface
~~~~~~~~~~~~~~~~~~~~
..  autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    Dataset.bnds.__getitem__
    Dataset.bnds.__contains__
    Dataset.bnds.__iter__
    Dataset.bnds.__len__
    Dataset.bnds.get
    Dataset.bnds.keys
    Dataset.bnds.items
    Dataset.bnds.values


DataArray
=========

The ``bnds`` accessor is not currently available for DataArray objects because it requires features of :py:mod:`xarray` that are still in development. See https://github.com/pydata/xarray/issues/1475 for more information.

..  attention::

    The ``bnds`` accessor is not currently available for DataArray objects. See https://github.com/pydata/xarray/issues/1475 for more information.


