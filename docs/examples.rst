.. _examples:

========
Examples
========

This section provides detailed examples of using correct_atmosphere for various
atmospheric correction tasks.

Example 1: Rayleigh Scattering Calculations
-------------------------------------------

Calculate Rayleigh optical properties across a range of wavelengths:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.rayleigh import (
        rayleigh_optical_thickness,
        geometric_air_mass_factor,
        rayleigh_depolarization_ratio,
        pressure_correction_coefficient,
        RayleighLUT,
    )

    # Calculate Rayleigh optical thickness across visible spectrum
    wavelengths = np.array([412, 443, 490, 510, 555, 670, 765, 865])
    tau_r = rayleigh_optical_thickness(wavelengths)

    print("Rayleigh Optical Thickness at Standard Pressure (1013.25 hPa):")
    for wl, tau in zip(wavelengths, tau_r):
        print(f"  {wl} nm: {tau:.4f}")

    # Effect of pressure on Rayleigh scattering
    pressures = [900, 950, 1000, 1013.25, 1030]
    print("\nRayleigh optical thickness at 443 nm for different pressures:")
    for p in pressures:
        tau = rayleigh_optical_thickness(443.0, pressure=p)
        print(f"  {p} hPa: {tau:.4f}")

    # Air mass factor for different geometries
    print("\nAir mass factor (M = 1/cos(theta_s) + 1/cos(theta_v)):")
    for sza in [0, 30, 45, 60]:
        M = geometric_air_mass_factor(sza, 20.0)
        print(f"  SZA={sza}deg, VZA=20deg: M={M:.3f}")

    # Synthetic Rayleigh reflectance calculation
    lut = RayleighLUT(sensor='seawifs')
    rho_r = lut.compute_synthetic(
        wavelength=443.0,
        solar_zenith=30.0,
        view_zenith=15.0,
        relative_azimuth=90.0,
    )
    print(f"\nSynthetic Rayleigh reflectance at 443 nm: {rho_r:.4f}")

Example 2: Gas Absorption Analysis
----------------------------------

Calculate gas absorption effects for O3 and NO2:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.gases import (
        ozone_transmittance,
        ozone_optical_thickness,
        no2_optical_thickness,
        gas_transmittance,
        interpolate_o3_cross_section,
        no2_correction_factor,
    )

    # Ozone transmittance vs wavelength
    wavelengths = np.array([412, 443, 490, 510, 555, 670])
    ozone_du = 350  # Dobson Units

    print(f"Ozone Transmittance (O3 = {ozone_du} DU, SZA=30, VZA=15):")
    for wl in wavelengths:
        t_o3 = ozone_transmittance(wl, ozone_du, 30.0, 15.0)
        print(f"  {wl} nm: {t_o3:.4f}")

    # Ozone transmittance vs ozone column
    print("\nOzone transmittance at 555 nm for different O3 columns:")
    for o3 in [200, 250, 300, 350, 400, 450]:
        t_o3 = ozone_transmittance(555.0, o3, 30.0, 15.0)
        print(f"  {o3} DU: {t_o3:.4f}")

    # NO2 correction factors
    no2_total = 2e16  # molecules/cm^2
    no2_strat = 1e16  # stratospheric (above 200m)

    corrections = no2_correction_factor(
        wavelength=443.0,
        no2_total=no2_total,
        no2_above_200m=no2_strat,
        solar_zenith=30.0,
        view_zenith=15.0,
    )
    print(f"\nNO2 correction factors at 443 nm:")
    print(f"  Path correction: {corrections['path_correction']:.4f}")
    print(f"  Solar path: {corrections['water_correction_solar']:.4f}")
    print(f"  View path: {corrections['water_correction_view']:.4f}")

    # Combined gas transmittance
    print("\nCombined gas transmittance (O3 + NO2):")
    for wl in wavelengths:
        t_gas = gas_transmittance(wl, 30.0, 15.0, ozone_du, 1.1e16)
        print(f"  {wl} nm: {t_gas:.4f}")

Example 3: Sun Glint Calculation
--------------------------------

Analyze sun glint patterns for different geometries and wind speeds:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.glint import (
        normalized_sun_glint,
        glint_mask,
        cox_munk_slope_variance,
        sun_glint_reflectance,
        two_path_transmittance,
    )
    from correct_atmosphere.constants import GLINT_THRESHOLD

    # Cox-Munk slope variance vs wind speed
    print("Cox-Munk Mean Square Slope vs Wind Speed:")
    for wind in [0, 2, 5, 10, 15, 20]:
        mss = cox_munk_slope_variance(wind)
        print(f"  {wind:2d} m/s: {mss:.4f}")

    # Normalized sun glint for different geometries
    print("\nNormalized Sun Glint L_GN [sr^-1]:")
    print("Wind speed = 10 m/s")
    print("       SZA=15  SZA=30  SZA=45  SZA=60")
    for phi in [0, 45, 90, 135, 180]:
        values = []
        for sza in [15, 30, 45, 60]:
            L_GN = normalized_sun_glint(sza, 15.0, phi, 10.0)
            values.append(L_GN)
        print(f"phi={phi:3d}  " + "  ".join(f"{v:.4f}" for v in values))

    # Glint masking analysis
    print(f"\nGlint threshold: {GLINT_THRESHOLD} sr^-1")
    print("Glint mask results (True = should mask):")
    for sza in [20, 30, 40]:
        for vza in [10, 20, 30]:
            for wind in [5, 10, 15]:
                masked = glint_mask(sza, vza, 45.0, wind)
                if masked:
                    L_GN = normalized_sun_glint(sza, vza, 45.0, wind)
                    print(f"  SZA={sza}, VZA={vza}, Wind={wind} m/s: "
                          f"L_GN={L_GN:.5f} - MASKED")

    # Sun glint reflectance with wavelength dependence
    print("\nSun glint reflectance at TOA for different wavelengths:")
    wavelengths = np.array([443, 555, 670, 865])
    rho_g = sun_glint_reflectance(30.0, 15.0, 90.0, 10.0, wavelengths)
    for wl, rho in zip(wavelengths, rho_g):
        print(f"  {wl} nm: {rho:.6f}")

Example 4: Whitecap Reflectance
-------------------------------

Calculate whitecap contributions for various conditions:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.whitecaps import (
        whitecap_fraction,
        whitecap_fraction_developed,
        whitecap_fraction_undeveloped,
        whitecap_reflectance,
        whitecap_spectral_factor,
        whitecap_toa_contribution,
    )
    from correct_atmosphere.constants import (
        WHITECAP_MIN_WIND_SPEED,
        WHITECAP_MAX_WIND_SPEED,
    )

    # Whitecap fraction vs wind speed
    print("Whitecap Fractional Coverage:")
    print("Wind (m/s)  Undeveloped   Developed")
    for wind in np.arange(4, 16, 1):
        f_undev = whitecap_fraction_undeveloped(wind)
        f_dev = whitecap_fraction_developed(wind)
        print(f"  {wind:5.1f}      {f_undev:.5f}      {f_dev:.5f}")

    # Note wind speed limits
    print(f"\nWhitecap correction applied for wind speeds: "
          f"{WHITECAP_MIN_WIND_SPEED} to {WHITECAP_MAX_WIND_SPEED} m/s")

    # Spectral dependence of whitecap reflectance
    wavelengths = np.array([412, 443, 490, 555, 670, 765, 865])
    print("\nWhitecap Spectral Factor (relative to blue):")
    factors = whitecap_spectral_factor(wavelengths)
    for wl, f in zip(wavelengths, factors):
        print(f"  {wl} nm: {f:.3f}")

    # Whitecap reflectance across spectrum
    wind_speed = 10.0
    print(f"\nWhitecap Reflectance at {wind_speed} m/s:")
    rho_wc = whitecap_reflectance(wind_speed, wavelengths)
    for wl, rho in zip(wavelengths, rho_wc):
        print(f"  {wl} nm: {rho:.6f}")

    # TOA contribution with diffuse transmittance
    t_sun = 0.90
    t_view = 0.92
    print(f"\nWhitecap contribution at TOA (t_sun={t_sun}, t_view={t_view}):")
    rho_wc_toa = whitecap_toa_contribution(wind_speed, wavelengths, t_sun, t_view)
    for wl, rho in zip(wavelengths, rho_wc_toa):
        print(f"  {wl} nm: {rho:.6f}")

Example 5: Aerosol Calculations
-------------------------------

Working with aerosol optical properties:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.aerosols import (
        angstrom_exponent,
        aerosol_optical_thickness,
        epsilon_ratio,
        AerosolLUT,
        AerosolModel,
        black_pixel_correction,
        should_apply_nonblack_pixel,
    )

    # Angstrom exponent from two-wavelength AOT
    tau_443 = 0.20
    tau_865 = 0.08
    alpha = angstrom_exponent(tau_443, tau_865, 443.0, 865.0)
    print(f"Angstrom exponent: {alpha:.2f}")
    print(f"  (tau_443={tau_443}, tau_865={tau_865})")

    # Extrapolate AOT to other wavelengths using Angstrom law
    print("\nExtrapolated AOT using Angstrom law:")
    wavelengths = np.array([412, 443, 490, 555, 670, 748, 869])
    for wl in wavelengths:
        tau = aerosol_optical_thickness(wl, tau_865, 865.0, alpha)
        print(f"  {wl} nm: {tau:.4f}")

    # Epsilon ratios for aerosol model selection
    print("\nEpsilon ratio = rho_a(lambda_1) / rho_a(lambda_2)")
    rho_748 = 0.020
    rho_869 = 0.015
    eps = epsilon_ratio(rho_748, rho_869)
    print(f"epsilon(748, 869) = {eps:.3f}")

    # Synthetic aerosol model epsilons
    lut = AerosolLUT('seawifs')
    print("\nSynthetic epsilon values for aerosol models (748/869 nm):")
    for model_id in range(10):
        eps = lut._synthetic_epsilon(model_id, 748.0, 865.0)
        print(f"  Model {model_id}: {eps:.3f}")

    # Non-black-pixel decision
    print("\nNon-black-pixel algorithm decision:")
    for chl in [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]:
        apply, weight = should_apply_nonblack_pixel(chl)
        status = "Always" if weight == 1.0 else ("Never" if weight == 0 else "Blend")
        print(f"  Chl={chl:.1f} mg/m³: {status} (weight={weight:.2f})")

Example 6: Transmittance Calculations
-------------------------------------

Direct and diffuse transmittance for various conditions:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.transmittance import (
        direct_transmittance,
        diffuse_transmittance,
        diffuse_transmittance_rayleigh,
        total_transmittance,
        two_path_direct_transmittance,
    )
    from correct_atmosphere.rayleigh import rayleigh_optical_thickness

    # Direct vs diffuse transmittance comparison
    print("Direct vs Diffuse Transmittance (tau=0.2, varying zenith angle):")
    print("Zenith    Direct   Diffuse")
    for theta in [0, 15, 30, 45, 60, 75]:
        T_direct = direct_transmittance(theta, 0.2)
        t_diffuse = diffuse_transmittance(theta, 550.0, aerosol_tau=0.1)
        print(f"  {theta:3d}deg   {T_direct:.4f}   {t_diffuse:.4f}")

    # Transmittance vs optical thickness
    print("\nDirect transmittance at nadir for varying optical thickness:")
    for tau in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        T = direct_transmittance(0.0, tau)
        print(f"  tau={tau:.1f}: T={T:.4f}")

    # Two-path transmittance for glint
    print("\nTwo-path direct transmittance (SZA=30, VZA=15):")
    wavelengths = np.array([443, 555, 670, 865])
    for wl in wavelengths:
        tau = rayleigh_optical_thickness(wl)
        T_two = two_path_direct_transmittance(30.0, 15.0, tau)
        print(f"  {wl} nm (tau_R={tau:.4f}): T_two={T_two:.4f}")

    # Total transmittance (both paths)
    print("\nTotal diffuse transmittance (sun + view paths):")
    for wl in wavelengths:
        t = total_transmittance(30.0, 15.0, wl, aerosol_tau=0.1)
        print(f"  {wl} nm: {t:.4f}")

Example 7: Normalization and BRDF Correction
--------------------------------------------

Converting water-leaving radiance to normalized products:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.normalization import (
        earth_sun_distance_correction,
        normalized_water_leaving_radiance,
        normalized_water_leaving_reflectance,
        remote_sensing_reflectance,
        rrs_to_normalized_reflectance,
        normalized_reflectance_to_rrs,
        BRDFCorrection,
        snell_angle,
    )

    # Earth-Sun distance correction over the year
    print("Earth-Sun Distance Correction Factor (R/R0)^2:")
    days = [1, 91, 172, 266, 365]  # Jan 1, Apr 1, Jun 21, Sep 23, Dec 31
    names = ["Jan 1", "Apr 1", "Jun 21", "Sep 23", "Dec 31"]
    for day, name in zip(days, names):
        corr = earth_sun_distance_correction(day)
        print(f"  Day {day:3d} ({name}): {corr:.4f}")

    # Remote-sensing reflectance from radiances
    Lw = 0.5  # mW cm^-2 um^-1 sr^-1
    Ed = 80.0  # mW cm^-2 um^-1
    Rrs = remote_sensing_reflectance(Lw, Ed)
    print(f"\nRemote-sensing reflectance: Rrs = {Rrs:.6f} sr^-1")

    # Conversion between Rrs and normalized reflectance
    rho_n = rrs_to_normalized_reflectance(Rrs)
    print(f"Normalized reflectance: [rho_w]_N = {rho_n:.6f}")
    Rrs_back = normalized_reflectance_to_rrs(rho_n)
    print(f"Back to Rrs: {Rrs_back:.6f} sr^-1")

    # Snell's law for in-water viewing angle
    print("\nSnell's law (air-to-water, n=1.34):")
    for theta_air in [0, 15, 30, 45, 60]:
        theta_water = snell_angle(theta_air)
        print(f"  {theta_air}deg in air -> {theta_water:.1f}deg in water")

    # BRDF correction factors
    brdf = BRDFCorrection()
    print("\nBRDF correction factors at 443 nm:")
    print("(Chl=0.5 mg/m³, phi=90°)")
    print("SZA   VZA   Factor")
    for sza in [0, 30, 45, 60]:
        for vza in [0, 15, 30]:
            factor = brdf.correction_factor(443.0, sza, vza, 90.0, 0.5)
            print(f" {sza:3d}   {vza:3d}   {factor:.4f}")

Example 8: Polarization Analysis
--------------------------------

Working with Stokes vectors and polarization correction:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.polarization import (
        StokesVector,
        rotation_matrix,
        rotate_stokes_vector,
        SensorPolarization,
        MuellerMatrix,
        compute_rayleigh_stokes,
        degree_of_polarization,
        polarization_correction_factor,
    )

    # Create and analyze a Stokes vector
    stokes = StokesVector(I=1.0, Q=0.15, U=0.08, V=0.0)
    print("Stokes Vector Analysis:")
    print(f"  I = {stokes.I}")
    print(f"  Q = {stokes.Q}")
    print(f"  U = {stokes.U}")
    print(f"  V = {stokes.V}")
    print(f"  Degree of polarization: {stokes.degree_of_polarization:.3f}")
    print(f"  Degree of linear polarization: {stokes.degree_of_linear_polarization:.3f}")
    print(f"  Reduced q = Q/I: {stokes.q:.3f}")
    print(f"  Reduced u = U/I: {stokes.u:.3f}")

    # Rotation of Stokes vector
    print("\nStokes vector rotation:")
    for alpha in [0, 45, 90, 135, 180]:
        rotated = rotate_stokes_vector(stokes, alpha)
        print(f"  alpha={alpha:3d}deg: Q'={rotated.Q:.4f}, U'={rotated.U:.4f}")

    # Sensor-specific polarization parameters
    print("\nSensor Polarization Sensitivity (m12):")
    for sensor in ['seawifs', 'modis_aqua', 'viirs_npp']:
        pol = SensorPolarization.from_sensor(sensor)
        print(f"  {sensor}:")
        for i, wl in enumerate(pol.wavelengths[:5]):
            print(f"    {wl} nm: m12={pol.m12[i]:.3f}, m13={pol.m13[i]:.3f}")

    # Mueller matrix for MODIS
    print("\nMODIS Aqua Mueller matrix elements:")
    for band in [412, 443, 667, 869]:
        M = MuellerMatrix.modis_aqua(band)
        print(f"  {band} nm: m12={M.m12:.3f}, m13={M.m13:.3f}")

    # Rayleigh Stokes vector computation
    print("\nRayleigh-scattered Stokes vector (443 nm):")
    stokes_r = compute_rayleigh_stokes(443.0, 30.0, 15.0, 90.0)
    print(f"  I = {stokes_r.I:.4f}")
    print(f"  Q = {stokes_r.Q:.4f}")
    print(f"  U = {stokes_r.U:.4f}")
    print(f"  DOP = {stokes_r.degree_of_linear_polarization:.3f}")

Example 9: Out-of-Band Correction
---------------------------------

Spectral response function and OOB correction:

.. code-block:: python

    import numpy as np
    from correct_atmosphere.outofband import (
        gaussian_srf,
        tophat_srf,
        band_averaged_radiance,
        apply_oob_correction,
        OOBCorrectionLUT,
        case1_rrs_spectrum,
        oob_correction_for_hyperspectral,
    )

    # Generate spectral response functions
    wavelengths = np.arange(400, 500, 1.0)

    print("Spectral Response Functions at 443 nm:")
    gaussian = gaussian_srf(wavelengths, center=443.0, fwhm=10.0)
    tophat = tophat_srf(wavelengths, center=443.0, width=10.0)
    print(f"  Gaussian peak: {max(gaussian):.4f} at {wavelengths[np.argmax(gaussian)]:.0f} nm")
    print(f"  Tophat coverage: {np.sum(tophat > 0)} nm")

    # Band-averaged radiance calculation
    # Simulated spectrum
    spectrum = 10.0 * np.exp(-((wavelengths - 450) / 30) ** 2)
    L_band = band_averaged_radiance(wavelengths, spectrum, gaussian)
    print(f"  Band-averaged radiance: {L_band:.4f}")

    # OOB correction for Rrs
    print("\nOut-of-Band Correction:")
    rrs = {412: 0.0080, 443: 0.0070, 490: 0.0050, 555: 0.0030, 670: 0.0010}
    rrs_corr = apply_oob_correction(rrs, sensor='seawifs')

    print("  Band    Original   Corrected   Change")
    for band in sorted(rrs.keys()):
        orig = rrs[band]
        corr = rrs_corr[band]
        change = (corr - orig) / orig * 100
        print(f"  {band}     {orig:.5f}    {corr:.5f}    {change:+.1f}%")

    # Case 1 water spectrum for different chlorophyll
    print("\nCase 1 Water Rrs Spectrum (simplified model):")
    wavelengths_vis = np.array([412, 443, 490, 555, 670])
    print("  Wavelength   Chl=0.1   Chl=1.0   Chl=5.0")
    for wl in wavelengths_vis:
        rrs_01 = case1_rrs_spectrum(wl, 0.1)
        rrs_10 = case1_rrs_spectrum(wl, 1.0)
        rrs_50 = case1_rrs_spectrum(wl, 5.0)
        print(f"    {wl:3d} nm    {rrs_01:.5f}   {rrs_10:.5f}   {rrs_50:.5f}")

Example 10: Constants and Sensor Configuration
----------------------------------------------

Accessing physical constants and sensor parameters:

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
        WHITECAP_REFLECTANCE_EFFECTIVE,
        SEAWIFS_BANDS,
        MODIS_AQUA_BANDS,
        VIIRS_BANDS,
        NIR_BANDS,
        get_sensor_bands,
        get_nir_bands,
        RAYLEIGH_TAU_SEAWIFS,
        O3_CROSS_SECTION_SEAWIFS,
        BRDF_CHL_VALUES,
    )

    # Physical constants
    print("Physical Constants:")
    print(f"  Standard pressure: {STANDARD_PRESSURE} hPa")
    print(f"  Standard temperature: {STANDARD_TEMPERATURE} K")
    print(f"  Mean Earth-Sun distance: {MEAN_EARTH_SUN_DISTANCE} AU")
    print(f"  Standard CO2: {STANDARD_CO2_PPM} ppm")
    print(f"  Water refractive index: {WATER_REFRACTIVE_INDEX}")

    # Algorithm thresholds
    print("\nAlgorithm Thresholds:")
    print(f"  Glint threshold: {GLINT_THRESHOLD} sr^-1")
    print(f"  Whitecap effective reflectance: {WHITECAP_REFLECTANCE_EFFECTIVE}")

    # Sensor bands
    print("\nSeaWiFS Band Definitions:")
    for band, (center, low, high) in SEAWIFS_BANDS.items():
        print(f"  {band}: {center} nm [{low}-{high}]")

    # NIR reference bands for aerosol correction
    print("\nNIR Reference Bands for Aerosol Correction:")
    for sensor, (band1, band2) in NIR_BANDS.items():
        print(f"  {sensor}: {band1}, {band2}")

    # Precomputed Rayleigh optical thickness
    print("\nPrecomputed Rayleigh Optical Thickness (SeaWiFS):")
    for band, tau in RAYLEIGH_TAU_SEAWIFS.items():
        print(f"  {band} nm: {tau:.4f}")

    # BRDF chlorophyll lookup values
    print(f"\nBRDF f/Q Table Chlorophyll Values: {BRDF_CHL_VALUES} mg/m³")
