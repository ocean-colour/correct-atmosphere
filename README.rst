==================
correct_atmosphere
==================

Atmospheric Correction for Satellite Ocean Color Radiometry
============================================================

A Python implementation of the NASA Ocean Biology Processing Group (OBPG)
atmospheric correction algorithms for ocean color remote sensing.

This package implements the algorithms documented in:

    Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
    *Atmospheric Correction for Satellite Ocean Color Radiometry*.
    NASA/TM-2016-217551.

Features
--------

* **Rayleigh Correction** (``rayleigh``): Molecular scattering with
  pressure corrections and lookup tables
* **Gas Absorption** (``gases``): O3 and NO2 absorption corrections
* **Sun Glint** (``glint``): Cox-Munk wave slope model for sun glint estimation
* **Whitecaps** (``whitecaps``): Wind-dependent whitecap reflectance
* **Aerosols** (``aerosols``): Black-pixel and iterative non-black-pixel algorithms
* **Transmittance** (``transmittance``): Direct and diffuse atmospheric transmittance
* **Normalization** (``normalization``): BRDF correction and Rrs computation
* **Polarization** (``polarization``): Stokes vectors and sensor polarization correction
* **Out-of-Band** (``outofband``): Spectral response function corrections
* **Constants** (``constants``): Physical constants and sensor band definitions

Requirements
------------

* Python >= 3.9
* NumPy >= 1.21
* SciPy >= 1.7
* xarray >= 0.19
* netCDF4 >= 1.5

Installation
------------

From PyPI (when available)::

    pip install correct_atmosphere

From source::

    git clone https://github.com/ocean-colour/correct-atmosphere.git
    cd correct-atmosphere
    pip install -e .

For development::

    pip install -e ".[dev,docs]"

Quick Start
-----------

Using individual modules:

.. code-block:: python

    from correct_atmosphere.rayleigh import rayleigh_optical_thickness
    from correct_atmosphere.gases import ozone_transmittance
    from correct_atmosphere.glint import normalized_sun_glint
    from correct_atmosphere.whitecaps import whitecap_reflectance

    # Rayleigh optical thickness at 443 nm
    tau_r = rayleigh_optical_thickness(443.0)
    print(f"Rayleigh optical thickness at 443 nm: {tau_r:.4f}")

    # Ozone transmittance
    t_o3 = ozone_transmittance(443.0, 350.0, 30.0, 15.0)
    print(f"O3 transmittance: {t_o3:.4f}")

    # Normalized sun glint
    L_GN = normalized_sun_glint(30.0, 15.0, 90.0, 10.0)
    print(f"Normalized sun glint: {L_GN:.6f} sr^-1")

    # Whitecap reflectance
    rho_wc = whitecap_reflectance(10.0, 550.0)
    print(f"Whitecap reflectance: {rho_wc:.6f}")

Using the full atmospheric correction pipeline:

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

    # TOA radiance array
    Lt = np.array([12.0, 11.0, 10.0, 8.0, 6.0, 5.0, 3.0, 2.5, 1.5, 1.2])

    # Process
    result = ac.process(Lt, geometry, ancillary)

    # Access results
    print(f"Rrs: {result.rrs}")
    print(f"Chlorophyll: {result.chlorophyll:.2f} mg/m³")

Supported Sensors
-----------------

* SeaWiFS (seawifs)
* MODIS-Aqua (modis_aqua)
* MODIS-Terra (modis_terra)
* VIIRS-NPP (viirs_npp)
* VIIRS-NOAA20 (viirs_noaa20)

Module Overview
---------------

**constants**: Physical constants and sensor configurations

.. code-block:: python

    from correct_atmosphere import STANDARD_PRESSURE, STANDARD_TEMPERATURE
    from correct_atmosphere.constants import get_sensor_bands, get_nir_bands

**rayleigh**: Rayleigh scattering calculations

.. code-block:: python

    from correct_atmosphere.rayleigh import (
        rayleigh_optical_thickness,
        geometric_air_mass_factor,
        RayleighLUT,
    )

**gases**: Gas absorption corrections

.. code-block:: python

    from correct_atmosphere.gases import (
        ozone_transmittance,
        gas_transmittance,
        no2_correction_factor,
    )

**glint**: Sun glint estimation

.. code-block:: python

    from correct_atmosphere.glint import (
        normalized_sun_glint,
        glint_mask,
        cox_munk_slope_variance,
    )

**whitecaps**: Whitecap reflectance

.. code-block:: python

    from correct_atmosphere.whitecaps import (
        whitecap_reflectance,
        whitecap_fraction,
    )

**aerosols**: Aerosol correction

.. code-block:: python

    from correct_atmosphere.aerosols import (
        angstrom_exponent,
        aerosol_optical_thickness,
        AerosolLUT,
    )

**transmittance**: Atmospheric transmittance

.. code-block:: python

    from correct_atmosphere.transmittance import (
        direct_transmittance,
        diffuse_transmittance,
        total_transmittance,
    )

**normalization**: BRDF and normalization

.. code-block:: python

    from correct_atmosphere.normalization import (
        remote_sensing_reflectance,
        earth_sun_distance_correction,
        BRDFCorrection,
    )

**polarization**: Polarization correction

.. code-block:: python

    from correct_atmosphere.polarization import (
        StokesVector,
        SensorPolarization,
        compute_rayleigh_stokes,
    )

**outofband**: Out-of-band correction

.. code-block:: python

    from correct_atmosphere.outofband import (
        apply_oob_correction,
        OOBCorrectionLUT,
    )

Documentation
-------------

Full documentation is available at https://correct_atmosphere.readthedocs.io

Theory and algorithm descriptions are based on the NASA Technical Memorandum
and the Ocean Optics Web Book (http://www.oceanopticsbook.info/).

Running Tests
-------------

.. code-block:: bash

    pytest

Contributing
------------

Contributions are welcome! Please see the contributing guidelines in the
documentation.

License
-------

This project is licensed under the MIT License - see the LICENSE file for
details.

References
----------

1. Mobley, C.D., et al. (2016). Atmospheric Correction for Satellite Ocean
   Color Radiometry. NASA/TM-2016-217551.

2. Gordon, H.R. and Wang, M. (1994). Retrieval of water-leaving radiance and
   aerosol optical thickness over the oceans with SeaWiFS: a preliminary
   algorithm. Applied Optics, 33:443-452.

3. Ahmad, Z., et al. (2010). New aerosol models for the retrieval of aerosol
   optical thickness and normalized water-leaving radiances from SeaWiFS and
   MODIS sensors. Applied Optics, 49:5545-5560.

4. Bailey, S.W., et al. (2010). Estimation of near-infrared water-leaving
   reflectance for satellite ocean color data processing. Optics Express,
   18:7521-7527.

5. Morel, A., et al. (2002). Bidirectional reflectance of oceanic waters:
   accounting for Raman emission and varying particle scattering phase
   function. Applied Optics, 41:6289-6306.
