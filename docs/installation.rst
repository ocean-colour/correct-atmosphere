.. _installation:

============
Installation
============

Requirements
------------

correct_atmosphere requires Python 3.9 or later and the following core dependencies:

* NumPy >= 1.21
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

    git clone https://github.com/ocean-colour/correct-atmosphere.git
    cd correct-atmosphere
    pip install -e .

Development Installation
------------------------

For development, install with extra dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This installs additional packages for testing:

* pytest >= 7.0.0
* pytest-cov >= 3.0.0
* black >= 22.0.0
* ruff >= 0.0.260
* mypy >= 0.950

For documentation, install the docs extras:

.. code-block:: bash

    pip install -e ".[docs]"

This installs:

* sphinx >= 4.0.0
* sphinx-rtd-theme >= 1.0.0
* numpydoc >= 1.2.0

Running Tests
-------------

Run the test suite using pytest:

.. code-block:: bash

    pytest

Or using setup.py:

.. code-block:: bash

    python setup.py pytest

Verifying Installation
----------------------

Verify the installation by running:

.. code-block:: python

    import correct_atmosphere
    print(correct_atmosphere.__version__)
    # Should print: 0.1.0

You can also verify the main components are importable:

.. code-block:: python

    from correct_atmosphere import AtmosphericCorrection
    from correct_atmosphere.rayleigh import rayleigh_optical_thickness
    from correct_atmosphere.whitecaps import whitecap_reflectance

    # Test basic functionality
    tau_r = rayleigh_optical_thickness(443.0)
    print(f"Rayleigh optical thickness at 443 nm: {tau_r:.4f}")
