.. _quickstart:

===========
Quick Start
===========

This guide demonstrates the basic usage of correct_atmosphere for atmospheric
correction of ocean color satellite data.

Basic Workflow
--------------

The typical workflow involves:

1. Loading TOA radiance or reflectance data
2. Gathering ancillary data (pressure, ozone, wind, etc.)
3. Running atmospheric correction
4. Analyzing water-leaving reflectance

Simple Example
--------------

.. code-block:: python

    from correct_atmosphere import AtmosphericCorrection
    
    # Initialize for MODIS-Aqua
    ac = AtmosphericCorrection(sensor='modis_aqua')
    
    # Example TOA reflectance (after calibration)
    rho_toa = {
        412: 0.12,
        443: 0.10,
        488: 0.08,
        531: 0.06,
        547: 0.055,
        667: 0.03,
        748: 0.02,
        869: 0.015,
    }
    
    # Run atmospheric correction
    result = ac.process(
        rho_toa=rho_toa,
        theta_s=30,        # Solar zenith angle (degrees)
        theta_v=20,        # Viewing zenith angle (degrees)
        phi=90,            # Relative azimuth (degrees)
        pressure=1013.25,  # Surface pressure (hPa)
        ozone_du=350,      # Ozone column (Dobson units)
        wind_speed=5,      # Wind speed (m/s)
    )
    
    # Extract results
    rrs = result['rrs']  # Remote-sensing reflectance
    print(f"Rrs(443) = {rrs[443]:.5f} sr⁻¹")

Using Individual Modules
------------------------

For more control, use individual correction modules:

.. code-block:: python

    from correct_atmosphere import rayleigh, gases, aerosols
    
    # Compute Rayleigh optical thickness
    tau_r = rayleigh.rayleigh_optical_thickness(443)
    print(f"τ_R(443) = {tau_r:.4f}")
    
    # Compute ozone transmittance
    t_o3 = gases.ozone_transmittance(
        wavelength=443,
        ozone_du=350,
        theta_s=30,
        theta_v=20,
    )
    print(f"t_O3(443) = {t_o3:.4f}")

Working with Satellite Data
---------------------------

For real satellite data, you'll typically:

1. Read Level-1B calibrated radiances
2. Extract ancillary data from auxiliary files
3. Process pixel-by-pixel or in batches

.. code-block:: python

    import numpy as np
    from correct_atmosphere import AtmosphericCorrection
    
    # Initialize
    ac = AtmosphericCorrection(sensor='viirs')
    
    # Process multiple pixels
    for i in range(npixels):
        result = ac.process(
            rho_toa=get_pixel_data(i),
            theta_s=solar_zenith[i],
            theta_v=view_zenith[i],
            phi=rel_azimuth[i],
            pressure=pressure[i],
            ozone_du=ozone[i],
            wind_speed=wind[i],
        )
        
        store_result(i, result)

Understanding the Output
------------------------

The ``process()`` method returns a dictionary containing:

* ``rrs``: Remote-sensing reflectance (sr⁻¹)
* ``rho_w``: Water-leaving reflectance (dimensionless)
* ``rho_a``: Aerosol reflectance
* ``tau_a``: Aerosol optical thickness
* ``flags``: Quality flags

.. code-block:: python

    result = ac.process(...)
    
    # Access components
    rrs = result['rrs']           # Dict by wavelength
    rho_aerosol = result['rho_a'] # Aerosol contribution
    aot_865 = result['tau_a'][865]  # AOT at 865 nm
    
    # Check quality
    if result['flags'].get('glint_warning'):
        print("Warning: possible glint contamination")

Next Steps
----------

* See :doc:`theory` for algorithm details
* See :doc:`api` for complete API reference
* See :doc:`examples` for more detailed examples
