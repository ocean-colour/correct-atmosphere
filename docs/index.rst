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
  water-leaving reflectance
* **Multi-sensor support**: SeaWiFS, MODIS-Aqua, MODIS-Terra, VIIRS-NPP
* **Modular design**: Use individual correction components or the full pipeline
* **Well-documented**: Extensive docstrings with equation references

Components
----------

The package includes modules for:

* **Rayleigh scattering** (Section 6.1): Molecular scattering corrections
* **Gas absorption** (Section 6.2): O₃ and NO₂ corrections
* **Sun glint** (Section 7): Specular reflection correction
* **Whitecaps** (Section 8): Foam reflectance
* **Aerosols** (Section 9): Path radiance estimation
* **Transmittance** (Section 4): Direct and diffuse transmission
* **Normalization** (Section 3): BRDF and f/Q corrections
* **Polarization** (Section 11): Sensor polarization sensitivity
* **Out-of-band** (Section 10): Spectral response corrections

Installation
------------

.. code-block:: bash

    pip install correct_atmosphere

For development:

.. code-block:: bash

    git clone https://github.com/username/correct_atmosphere.git
    cd correct_atmosphere
    pip install -e ".[dev]"

Basic Usage
-----------

.. code-block:: python

    from correct_atmosphere import AtmosphericCorrection
    
    # Initialize for a specific sensor
    ac = AtmosphericCorrection(sensor='modis_aqua')
    
    # Process TOA reflectance
    result = ac.process(
        rho_toa={412: 0.12, 443: 0.10, 490: 0.08, ...},
        theta_s=30,      # Solar zenith angle
        theta_v=30,      # Viewing zenith angle
        phi=90,          # Relative azimuth
        pressure=1013,   # Sea-level pressure (hPa)
        ozone_du=350,    # Ozone (Dobson units)
        wind_speed=5,    # Wind speed (m/s)
    )
    
    # Access water-leaving reflectance
    rrs = result['rrs']

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
