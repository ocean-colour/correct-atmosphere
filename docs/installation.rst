.. _installation:

============
Installation
============

Requirements
------------

correct_atmosphere requires Python 3.9 or later and the following dependencies:

* NumPy >= 1.20
* SciPy >= 1.7
* xarray >= 0.19
* netCDF4 >= 1.5

Installation from PyPI
----------------------

The simplest way to install correct_atmosphere is using pip:

.. code-block:: bash

    pip install correct_atmosphere

Installation from Source
------------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/username/correct_atmosphere.git
    cd correct_atmosphere
    pip install -e .

Development Installation
------------------------

For development, install with extra dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This installs additional packages for testing and documentation:

* pytest
* black
* ruff
* mypy
* sphinx
* sphinx-rtd-theme
* numpydoc

Verifying Installation
----------------------

Verify the installation by running:

.. code-block:: python

    import correct_atmosphere
    print(correct_atmosphere.__version__)

Or run the test suite:

.. code-block:: bash

    pytest tests/

Optional Dependencies
---------------------

For additional functionality:

* **matplotlib**: For visualization examples
* **cartopy**: For map projections in examples
* **h5py**: For reading HDF5 files
