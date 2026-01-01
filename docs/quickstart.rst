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

Using Individual Modules
------------------------

The package provides individual correction modules for fine-grained control.
Here are examples using the implemented functions:

Rayleigh Scattering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.rayleigh import (
        rayleigh_optical_thickness,
        geometric_air_mass_factor,
        rayleigh_depolarization_ratio,
    )

    # Compute Rayleigh optical thickness at 443 nm
    tau_r = rayleigh_optical_thickness(443.0)
    print(f"Rayleigh optical thickness at 443 nm: {tau_r:.4f}")

    # With pressure correction
    tau_r_pressure = rayleigh_optical_thickness(443.0, pressure=1000.0)
    print(f"At 1000 hPa: {tau_r_pressure:.4f}")

    # Compute air mass factor
    M = geometric_air_mass_factor(solar_zenith=30.0, view_zenith=15.0)
    print(f"Air mass factor: {M:.3f}")

Gas Absorption
~~~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.gases import (
        ozone_transmittance,
        gas_transmittance,
        ozone_optical_thickness,
    )

    # Compute ozone transmittance
    t_o3 = ozone_transmittance(
        wavelength=443.0,
        o3_concentration=350.0,  # Dobson Units
        solar_zenith=30.0,
        view_zenith=15.0,
    )
    print(f"O3 transmittance at 443 nm: {t_o3:.4f}")

    # Combined gas transmittance (O3 + NO2)
    t_gas = gas_transmittance(
        wavelength=443.0,
        solar_zenith=30.0,
        view_zenith=15.0,
        o3_concentration=350.0,
        no2_concentration=1.1e16,
    )
    print(f"Total gas transmittance: {t_gas:.4f}")

Sun Glint
~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.glint import (
        normalized_sun_glint,
        glint_mask,
        sun_glint_reflectance,
        cox_munk_slope_variance,
    )

    # Calculate normalized sun glint
    L_GN = normalized_sun_glint(
        solar_zenith=30.0,
        view_zenith=15.0,
        relative_azimuth=90.0,
        wind_speed=10.0,
    )
    print(f"Normalized sun glint: {L_GN:.6f} sr^-1")

    # Check if pixel should be masked
    masked = glint_mask(30.0, 15.0, 90.0, 10.0)
    print(f"Should mask: {masked}")

    # Cox-Munk mean square slope
    mss = cox_munk_slope_variance(10.0)
    print(f"Mean square slope at 10 m/s: {mss:.4f}")

Whitecaps
~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.whitecaps import (
        whitecap_reflectance,
        whitecap_fraction,
        whitecap_spectral_factor,
    )

    # Calculate whitecap reflectance
    rho_wc = whitecap_reflectance(wind_speed=10.0, wavelength=550.0)
    print(f"Whitecap reflectance at 10 m/s, 550 nm: {rho_wc:.6f}")

    # Fractional coverage
    frac = whitecap_fraction(10.0)
    print(f"Whitecap fraction at 10 m/s: {frac:.4f}")

    # Spectral factor
    import numpy as np
    wavelengths = np.array([412, 555, 670, 865])
    factors = whitecap_spectral_factor(wavelengths)
    print(f"Spectral factors: {factors}")

Transmittance
~~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.transmittance import (
        direct_transmittance,
        diffuse_transmittance,
        total_transmittance,
    )

    # Direct transmittance
    T = direct_transmittance(zenith_angle=30.0, optical_thickness=0.2)
    print(f"Direct transmittance: {T:.4f}")

    # Diffuse transmittance
    t = diffuse_transmittance(
        zenith_angle=30.0,
        wavelength=550.0,
        aerosol_tau=0.1,
    )
    print(f"Diffuse transmittance: {t:.4f}")

    # Total (two-path) transmittance
    t_total = total_transmittance(
        solar_zenith=30.0,
        view_zenith=15.0,
        wavelength=550.0,
        aerosol_tau=0.1,
    )
    print(f"Total transmittance: {t_total:.4f}")

Aerosols
~~~~~~~~

.. code-block:: python

    from correct_atmosphere.aerosols import (
        angstrom_exponent,
        aerosol_optical_thickness,
        epsilon_ratio,
    )

    # Calculate Angstrom exponent
    alpha = angstrom_exponent(
        tau_1=0.15,
        tau_2=0.10,
        wavelength_1=443.0,
        wavelength_2=865.0,
    )
    print(f"Angstrom exponent: {alpha:.2f}")

    # Extrapolate AOT to other wavelengths
    tau_443 = aerosol_optical_thickness(
        wavelength=443.0,
        reference_tau=0.10,
        reference_wavelength=865.0,
        alpha=alpha,
    )
    print(f"AOT at 443 nm: {tau_443:.3f}")

Normalization and BRDF
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.normalization import (
        remote_sensing_reflectance,
        normalized_water_leaving_radiance,
        earth_sun_distance_correction,
        BRDFCorrection,
    )

    # Remote-sensing reflectance
    Rrs = remote_sensing_reflectance(lw=0.5, ed=500.0)
    print(f"Rrs: {Rrs:.6f} sr^-1")

    # Earth-Sun distance correction
    correction = earth_sun_distance_correction(day_of_year=172)
    print(f"Earth-Sun distance correction (summer): {correction:.4f}")

    # BRDF correction factor
    brdf = BRDFCorrection()
    factor = brdf.correction_factor(
        wavelength=443.0,
        solar_zenith=30.0,
        view_zenith=15.0,
        relative_azimuth=90.0,
        chlorophyll=0.5,
    )
    print(f"BRDF correction factor: {factor:.4f}")

Polarization
~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.polarization import (
        StokesVector,
        rotation_matrix,
        SensorPolarization,
        compute_rayleigh_stokes,
    )

    # Create a Stokes vector
    stokes = StokesVector(I=1.0, Q=0.1, U=0.05, V=0.0)
    print(f"Degree of polarization: {stokes.degree_of_polarization:.3f}")

    # Get sensor polarization parameters
    sensor_pol = SensorPolarization.from_sensor('modis_aqua')
    print(f"Sensor wavelengths: {sensor_pol.wavelengths}")

    # Compute Rayleigh Stokes vector
    stokes_r = compute_rayleigh_stokes(
        wavelength=443.0,
        solar_zenith=30.0,
        view_zenith=15.0,
        relative_azimuth=90.0,
    )
    print(f"Rayleigh DOP: {stokes_r.degree_of_linear_polarization:.3f}")

Out-of-Band Correction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from correct_atmosphere.outofband import (
        apply_oob_correction,
        gaussian_srf,
        band_averaged_radiance,
    )

    # Apply OOB correction to Rrs values
    rrs = {412: 0.008, 443: 0.007, 490: 0.005, 555: 0.003, 670: 0.001}
    rrs_corrected = apply_oob_correction(rrs, sensor='seawifs')
    print(f"Corrected Rrs(443): {rrs_corrected[443]:.6f}")

Using the AtmosphericCorrection Class
-------------------------------------

For full atmospheric correction, use the ``AtmosphericCorrection`` class:

.. code-block:: python

    import numpy as np
    from correct_atmosphere import AtmosphericCorrection
    from correct_atmosphere.correction import (
        GeometryAngles,
        AncillaryData,
    )

    # Initialize for a sensor
    ac = AtmosphericCorrection(sensor='modis_aqua')
    print(f"Sensor wavelengths: {ac.wavelengths}")

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
        relative_humidity=80.0,
    )

    # TOA radiance (example values)
    Lt = np.array([12.0, 11.0, 10.0, 8.0, 6.0, 5.0, 3.0, 2.5, 1.5, 1.2])

    # Process (requires LUT data which is not yet fully implemented)
    # result = ac.process(Lt, geometry, ancillary)

Understanding the Output
------------------------

The ``process()`` method returns a ``CorrectionResult`` dataclass containing:

* ``rrs``: Remote-sensing reflectance Rrs [sr^-1]
* ``nLw``: Normalized water-leaving radiance [mW cm^-2 um^-1 sr^-1]
* ``rho_w``: Exact normalized water-leaving reflectance
* ``Lw``: Water-leaving radiance at sea surface
* ``La``: Aerosol path radiance
* ``taua``: Aerosol optical thickness at each wavelength
* ``angstrom``: Angstrom exponent for retrieved aerosol
* ``chlorophyll``: Estimated chlorophyll concentration [mg/m^3]
* ``flags``: Quality and status flags (``CorrectionFlags`` dataclass)

Physical Constants
------------------

The package provides standard physical constants:

.. code-block:: python

    from correct_atmosphere import (
        STANDARD_PRESSURE,
        STANDARD_TEMPERATURE,
        MEAN_EARTH_SUN_DISTANCE,
    )
    from correct_atmosphere.constants import (
        STANDARD_CO2_PPM,
        WATER_REFRACTIVE_INDEX,
        GLINT_THRESHOLD,
        get_sensor_bands,
        get_nir_bands,
    )

    print(f"Standard pressure: {STANDARD_PRESSURE} hPa")
    print(f"Glint threshold: {GLINT_THRESHOLD} sr^-1")

    # Get sensor-specific band info
    bands = get_sensor_bands('MODIS-Aqua')
    nir_bands = get_nir_bands('MODIS-Aqua')
    print(f"NIR bands for aerosol: {nir_bands}")

Next Steps
----------

* See :doc:`theory` for algorithm details
* See :doc:`api` for complete API reference
* See :doc:`examples` for more detailed examples
