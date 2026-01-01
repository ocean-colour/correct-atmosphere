.. correct_atmosphere documentation master file

=========================================
correct_atmosphere: Atmospheric Correction
=========================================

**correct_atmosphere** is a Python implementation of the NASA Ocean Biology Processing
Group (OBPG) atmospheric correction algorithms for satellite ocean color
remote sensing.

This package implements the algorithms documented in:

    Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
    *Atmospheric Correction for Satellite Ocean Color Radiometry*.
    NASA Technical Memorandum 2016-217551.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   theory
   api
   examples

Quick Links
-----------

* :ref:`Installation <installation>`
* :ref:`Quick Start Guide <quickstart>`
* :ref:`API Reference <api>`
* :doc:`Theory and Background <theory>`

Features
--------

* **Complete atmospheric correction pipeline**: From TOA radiance to
  water-leaving reflectance via the ``AtmosphericCorrection`` class
* **Multi-sensor support**: SeaWiFS, MODIS-Aqua, MODIS-Terra, VIIRS-NPP, VIIRS-NOAA20
* **Modular design**: Use individual correction components or the full pipeline
* **Well-documented**: Extensive docstrings with equation references to NASA TM-2016-217551
* **Physical constants**: Comprehensive set of constants for atmospheric correction

Implemented Modules
-------------------

The package includes the following modules:

* **constants** - Physical constants, sensor band definitions, and algorithm thresholds
* **rayleigh** - Rayleigh scattering optical thickness and reflectance (Section 6.1)
* **gases** - O3 and NO2 gas absorption corrections (Section 6.2)
* **glint** - Sun glint estimation and masking using Cox-Munk model (Section 7)
* **whitecaps** - Whitecap/foam reflectance contributions (Section 8)
* **aerosols** - Aerosol path radiance and optical thickness (Section 9)
* **transmittance** - Direct and diffuse atmospheric transmittance (Section 4)
* **normalization** - BRDF correction, Earth-Sun distance, Rrs computation (Section 3)
* **polarization** - Stokes vectors and sensor polarization correction (Section 11)
* **outofband** - Out-of-band spectral response correction (Section 10)
* **correction** - Main ``AtmosphericCorrection`` class integrating all components

Installation
------------

.. code-block:: bash

    pip install correct_atmosphere

For development:

.. code-block:: bash

    git clone https://github.com/ocean-colour/correct-atmosphere.git
    cd correct-atmosphere
    pip install -e ".[dev]"

Basic Usage
-----------

Using individual modules:

.. code-block:: python

    from correct_atmosphere.rayleigh import rayleigh_optical_thickness
    from correct_atmosphere.gases import ozone_transmittance
    from correct_atmosphere.glint import normalized_sun_glint
    from correct_atmosphere.whitecaps import whitecap_reflectance

    # Rayleigh optical thickness at 443 nm
    tau_r = rayleigh_optical_thickness(443.0)
    print(f"Rayleigh optical thickness: {tau_r:.4f}")

    # Ozone transmittance
    t_o3 = ozone_transmittance(443.0, 350.0, 30.0, 15.0)
    print(f"O3 transmittance: {t_o3:.4f}")

    # Sun glint
    L_GN = normalized_sun_glint(30.0, 15.0, 90.0, 10.0)
    print(f"Normalized sun glint: {L_GN:.6f} sr^-1")

    # Whitecap reflectance
    rho_wc = whitecap_reflectance(10.0, 550.0)
    print(f"Whitecap reflectance: {rho_wc:.6f}")

Using the full atmospheric correction processor:

.. code-block:: python

    import numpy as np
    from correct_atmosphere import AtmosphericCorrection
    from correct_atmosphere.correction import GeometryAngles, AncillaryData

    # Initialize for MODIS-Aqua
    ac = AtmosphericCorrection(sensor='modis_aqua')

    # Set up geometry
    geometry = GeometryAngles(
        solar_zenith=30.0,
        solar_azimuth=135.0,
        view_zenith=15.0,
        view_azimuth=90.0,
    )

    # Set up ancillary data
    ancillary = AncillaryData(
        pressure=1013.25,
        wind_speed=5.0,
        ozone=300.0,
    )

    # TOA radiance array (n_bands,)
    Lt = np.array([12.0, 11.0, 10.0, ...])

    # Process
    result = ac.process(Lt, geometry, ancillary)

    # Access results
    print(f"Rrs: {result.rrs}")
    print(f"Chlorophyll: {result.chlorophyll:.2f} mg/m³")

Constants and Configuration
---------------------------

.. code-block:: python

    from correct_atmosphere import (
        STANDARD_PRESSURE,
        STANDARD_TEMPERATURE,
        MEAN_EARTH_SUN_DISTANCE,
    )
    from correct_atmosphere.constants import (
        GLINT_THRESHOLD,
        get_sensor_bands,
        get_nir_bands,
    )

    print(f"Standard pressure: {STANDARD_PRESSURE} hPa")
    print(f"Glint threshold: {GLINT_THRESHOLD} sr^-1")

    # Get sensor configuration
    bands = get_sensor_bands('MODIS-Aqua')
    nir = get_nir_bands('MODIS-Aqua')

Requirements
------------

* Python >= 3.9
* NumPy >= 1.21
* SciPy >= 1.7
* xarray >= 0.19
* netCDF4 >= 1.5

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
