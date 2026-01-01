===========
oceanatmos
===========

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

* **Rayleigh Correction**: Scattering by atmospheric gas molecules with
  wind speed and pressure effects
* **Gas Absorption**: Corrections for O₃ and NO₂ absorption
* **Sun Glint**: Direct sun glint correction using Cox-Munk wave slope
  distribution
* **Whitecaps**: Wind speed-dependent whitecap reflectance
* **Aerosol Correction**: Black-pixel and non-black-pixel algorithms for
  aerosol path radiance estimation
* **Normalized Reflectances**: Calculation of exact normalized water-leaving
  reflectance with BRDF corrections
* **Transmittance**: Direct and diffuse atmospheric transmittance calculations
* **Polarization Correction**: Sensor polarization sensitivity correction
* **Out-of-Band Correction**: Spectral band response corrections

Installation
------------

From PyPI (when available)::

    pip install oceanatmos

From source::

    git clone https://github.com/yourusername/oceanatmos.git
    cd oceanatmos
    pip install -e .

For development::

    pip install -e ".[dev,docs]"

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from oceanatmos import AtmosphericCorrection
    from oceanatmos.rayleigh import rayleigh_reflectance
    from oceanatmos.transmittance import diffuse_transmittance

    # Calculate Rayleigh optical thickness at 443 nm
    from oceanatmos.rayleigh import rayleigh_optical_thickness
    tau_r = rayleigh_optical_thickness(443.0)  # nm
    print(f"Rayleigh optical thickness at 443 nm: {tau_r:.4f}")

    # Calculate whitecap reflectance for 10 m/s wind
    from oceanatmos.whitecaps import whitecap_reflectance
    rho_wc = whitecap_reflectance(wind_speed=10.0, wavelength=550.0)
    print(f"Whitecap reflectance: {rho_wc:.6f}")

    # Full atmospheric correction pipeline
    ac = AtmosphericCorrection(sensor='MODIS-Aqua')
    
    # Load TOA radiance data (example)
    # result = ac.process(Lt, geometry, ancillary)

Supported Sensors
-----------------

* SeaWiFS
* MODIS-Aqua
* MODIS-Terra
* VIIRS-NPP
* VIIRS-NOAA20

Documentation
-------------

Full documentation is available at https://oceanatmos.readthedocs.io

Theory and algorithm descriptions are based on the NASA Technical Memorandum
and the Ocean Optics Web Book (http://www.oceanopticsbook.info/).

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
