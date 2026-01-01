.. _examples:

========
Examples
========

This section provides detailed examples of using oceanatmos for various
atmospheric correction tasks.

Example 1: Basic Atmospheric Correction
---------------------------------------

Complete example processing a single pixel:

.. code-block:: python

    from oceanatmos import AtmosphericCorrection
    
    # Initialize for SeaWiFS
    ac = AtmosphericCorrection(sensor='seawifs')
    
    # TOA reflectance from Level-1B data
    rho_toa = {
        412: 0.1188,
        443: 0.0989,
        490: 0.0788,
        510: 0.0686,
        555: 0.0507,
        670: 0.0281,
        765: 0.0193,
        865: 0.0147,
    }
    
    # Geometry and ancillary data
    result = ac.process(
        rho_toa=rho_toa,
        theta_s=32.5,      # Solar zenith angle
        theta_v=25.3,      # Viewing zenith angle
        phi=112.7,         # Relative azimuth
        pressure=1010.5,   # Surface pressure (hPa)
        ozone_du=285,      # Ozone column (DU)
        no2_conc=8.5e15,   # NO2 column (molecules/cm²)
        wind_speed=6.2,    # Wind speed (m/s)
        relative_humidity=78,  # Relative humidity (%)
    )
    
    # Print results
    print("Water-leaving reflectance:")
    for band, rrs in sorted(result['rrs'].items()):
        print(f"  Rrs({band}) = {rrs:.6f} sr⁻¹")
    
    print(f"\nAerosol optical thickness (865): {result['tau_a'][865]:.3f}")

Example 2: Processing with Individual Modules
---------------------------------------------

Using modules for step-by-step correction:

.. code-block:: python

    from oceanatmos import rayleigh, gases, glint, whitecaps, aerosols
    from oceanatmos import normalization
    import numpy as np
    
    # Input parameters
    wavelength = 443  # nm
    theta_s = 30      # Solar zenith
    theta_v = 20      # View zenith
    phi = 90          # Relative azimuth
    rho_toa = 0.10    # TOA reflectance
    
    # Step 1: Gas transmittance correction
    t_gas = gases.gas_transmittance(
        wavelength=wavelength,
        ozone_du=350,
        no2_conc=1e16,
        theta_s=theta_s,
        theta_v=theta_v,
    )
    rho_corr = rho_toa / t_gas
    print(f"After gas correction: {rho_corr:.4f}")
    
    # Step 2: Remove Rayleigh contribution
    tau_r = rayleigh.rayleigh_optical_thickness(wavelength)
    rho_r = rayleigh.RayleighLUT(sensor='seawifs').compute_synthetic(
        wavelength, theta_s, theta_v, phi, wind_speed=5
    )
    rho_corr = rho_corr - rho_r
    print(f"After Rayleigh removal: {rho_corr:.4f}")
    
    # Step 3: Remove whitecap contribution
    rho_wc = whitecaps.whitecap_reflectance(wavelength, wind_speed=8)
    t_diffuse = 0.90  # Approximate
    rho_corr = rho_corr - t_diffuse * rho_wc
    print(f"After whitecap removal: {rho_corr:.4f}")

Example 3: Batch Processing
---------------------------

Processing multiple pixels efficiently:

.. code-block:: python

    import numpy as np
    from oceanatmos import AtmosphericCorrection
    
    # Initialize
    ac = AtmosphericCorrection(sensor='modis_aqua')
    
    # Simulate data for 1000 pixels
    npixels = 1000
    np.random.seed(42)
    
    # Random TOA reflectances (example)
    rho_toa_array = {
        band: 0.05 + 0.1 * np.random.rand(npixels) * (865/band)**0.5
        for band in [412, 443, 488, 547, 667, 748, 869]
    }
    
    # Random geometry
    theta_s = 20 + 40 * np.random.rand(npixels)
    theta_v = 30 * np.random.rand(npixels)
    phi = 360 * np.random.rand(npixels)
    
    # Process
    results = []
    for i in range(npixels):
        pixel_toa = {band: arr[i] for band, arr in rho_toa_array.items()}
        
        result = ac.process(
            rho_toa=pixel_toa,
            theta_s=theta_s[i],
            theta_v=theta_v[i],
            phi=phi[i],
            pressure=1013,
            ozone_du=350,
            wind_speed=5,
        )
        results.append(result)
    
    # Analyze
    rrs_443 = [r['rrs'].get(443, np.nan) for r in results]
    print(f"Mean Rrs(443): {np.nanmean(rrs_443):.5f} sr⁻¹")

Example 4: Comparing Sensors
----------------------------

Processing the same scene with different sensor configurations:

.. code-block:: python

    from oceanatmos import AtmosphericCorrection
    
    # Common TOA reflectances (approximate band matching)
    common_toa = {
        'blue': 0.12,
        'green': 0.06,
        'red': 0.03,
        'nir': 0.015,
    }
    
    # SeaWiFS bands
    seawifs_toa = {443: common_toa['blue'], 555: common_toa['green'],
                   670: common_toa['red'], 865: common_toa['nir']}
    
    # MODIS bands
    modis_toa = {443: common_toa['blue'], 547: common_toa['green'],
                 667: common_toa['red'], 869: common_toa['nir']}
    
    # Common geometry and ancillary
    kwargs = dict(theta_s=30, theta_v=20, phi=90,
                  pressure=1013, ozone_du=350, wind_speed=5)
    
    # Process with each sensor
    ac_sw = AtmosphericCorrection(sensor='seawifs')
    ac_mod = AtmosphericCorrection(sensor='modis_aqua')
    
    result_sw = ac_sw.process(rho_toa=seawifs_toa, **kwargs)
    result_mod = ac_mod.process(rho_toa=modis_toa, **kwargs)
    
    print("SeaWiFS Rrs(443):", result_sw['rrs'][443])
    print("MODIS Rrs(443):", result_mod['rrs'][443])
