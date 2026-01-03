.. _theory:

========================
Theory and Background
========================

This section provides an overview of the atmospheric correction algorithms
implemented in correct_atmosphere, based on NASA TM-2016-217551 (Mobley et al., 2016).

The Atmospheric Correction Problem
==================================

Ocean color remote sensing aims to retrieve the water-leaving radiance
:math:`L_w`, which carries information about ocean constituents (chlorophyll,
suspended particles, etc.). However, satellites measure top-of-atmosphere (TOA)
radiance :math:`L_t`, which includes:

.. math::

    L_t = L_R + L_A + T L_g + t L_{wc} + t L_w

where:

* :math:`L_R` - Rayleigh (molecular) scattering
* :math:`L_A` - Aerosol scattering (including aerosol-Rayleigh interaction)
* :math:`L_g` - Sun glint
* :math:`L_{wc}` - Whitecap reflectance
* :math:`L_w` - Water-leaving radiance
* :math:`T, t` - Direct and diffuse transmittances

The water-leaving radiance is typically only 5-10% of the TOA signal,
making accurate atmospheric correction essential.

**Implementation**: The main ``AtmosphericCorrection`` class in
``correct_atmosphere.correction`` implements the full algorithm.

Rayleigh Scattering
===================

Molecular scattering by atmospheric gases follows the Rayleigh model.
The Rayleigh optical thickness at sea-level pressure is given by
Bodhaine et al. (1999):

.. math::

    \tau_R(\lambda) = 0.0021520 \frac{1.0455996 - 341.29061\lambda^{-2} - 0.90230850\lambda^2}
                                      {1.0 + 0.0027059889\lambda^{-2} - 85.968563\lambda^2}

The pressure correction follows Wang (2005):

.. math::

    \tau_R(P) = \tau_R(P_0) \cdot \frac{P}{P_0}

**Implementation**: The ``correct_atmosphere.rayleigh`` module provides:

* ``rayleigh_optical_thickness(wavelength, pressure)`` - Compute :math:`\tau_R` with optional pressure correction
* ``geometric_air_mass_factor(solar_zenith, view_zenith)`` - Compute :math:`M = 1/\cos\theta_s + 1/\cos\theta_v`
* ``rayleigh_depolarization_ratio(wavelength)`` - King factor for depolarization
* ``RayleighLUT`` - Lookup table interpolation for Rayleigh reflectance

Gas Absorption
==============

Ozone
-----

Ozone absorption transmittance (Section 6.2.1 of TM-2016-217551):

.. math::

    t_{O_3} = \exp\left(-\tau_{O_3} M\right)

where :math:`M = 1/\cos\theta_s + 1/\cos\theta_v` is the geometric air mass
factor and :math:`\tau_{O_3} = [O_3] \cdot k_{O_3}(\lambda)`.

The ozone cross-section :math:`k_{O_3}(\lambda)` varies strongly in the
Chappuis band (400-700 nm), with maximum absorption near 600 nm.

**Implementation**: The ``correct_atmosphere.gases`` module provides:

* ``ozone_transmittance(wavelength, o3_concentration, solar_zenith, view_zenith)``
* ``ozone_optical_thickness(wavelength, o3_concentration)``
* ``interpolate_o3_cross_section(wavelength, sensor)`` - Sensor-specific cross-sections

NO₂
---

NO₂ correction follows Ahmad et al. (2007), accounting for the vertical
distribution of NO₂ in the troposphere (Section 6.2.2). The correction accounts
for the fact that NO₂ resides primarily in the troposphere, below most of the
Rayleigh scattering.

**Implementation**:

* ``no2_optical_thickness(wavelength, no2_concentration)``
* ``no2_correction_factor(wavelength, no2_total, no2_above_200m, solar_zenith, view_zenith)``
* ``gas_transmittance(...)`` - Combined O₃ and NO₂ transmittance

Sun Glint
=========

Sun glint is computed using the Cox-Munk wave slope distribution (Section 7):

.. math::

    \sigma^2 = 0.00512 \cdot U + 0.003

where :math:`U` is wind speed in m/s. The probability distribution of wave
facets with appropriate slopes determines the glint radiance.

The normalized sun glint :math:`L_{GN}` represents the glint radiance normalized
by the Fresnel reflectance and irradiance. Pixels with :math:`L_{GN} > 0.005` sr⁻¹
are typically masked.

**Implementation**: The ``correct_atmosphere.glint`` module provides:

* ``cox_munk_slope_variance(wind_speed)`` - Mean square slope :math:`\sigma^2`
* ``normalized_sun_glint(solar_zenith, view_zenith, relative_azimuth, wind_speed)``
* ``glint_mask(...)`` - Boolean mask for high-glint pixels
* ``sun_glint_reflectance(...)`` - Glint contribution at TOA
* ``two_path_transmittance(...)`` - Direct transmittance for glint

Whitecaps
=========

Whitecap/foam contributions are significant at high wind speeds (Section 8).
The fractional coverage for undeveloped seas (Stramska & Petelski, 2003):

.. math::

    F_{wc} = 8.75 \times 10^{-5} (U - 6.33)^3

Applied for wind speeds 6.33-12 m/s. For developed seas, a lower whitecap
fraction is used.

The whitecap reflectance has a spectral dependence, decreasing toward the NIR
due to water absorption within the foam.

**Implementation**: The ``correct_atmosphere.whitecaps`` module provides:

* ``whitecap_fraction(wind_speed)`` - Fractional coverage :math:`F_{wc}`
* ``whitecap_fraction_undeveloped(wind_speed)`` - For undeveloped seas
* ``whitecap_fraction_developed(wind_speed)`` - For fully developed seas
* ``whitecap_reflectance(wind_speed, wavelength)`` - Total whitecap reflectance
* ``whitecap_spectral_factor(wavelength)`` - Spectral dependence
* ``whitecap_toa_contribution(...)`` - Whitecap contribution at TOA

Aerosol Correction
==================

The aerosol correction is the most challenging step (Section 9). The algorithm:

1. **Black-pixel assumption**: Assume :math:`L_w = 0` at NIR wavelengths
   (valid for oligotrophic waters)
2. **Model selection**: Match measured :math:`\epsilon(\lambda_1, \lambda_2)`
   to aerosol model lookup tables
3. **Extrapolation**: Extrapolate aerosol reflectance to visible wavelengths
   using the Angstrom relationship

The epsilon ratio characterizes aerosol spectral behavior:

.. math::

    \epsilon(\lambda_1, \lambda_2) = \frac{\rho_A(\lambda_1)}{\rho_A(\lambda_2)}

For waters with non-negligible NIR reflectance (chlorophyll > 0.3 mg/m³),
the iterative non-black-pixel algorithm (Bailey et al., 2010) estimates
:math:`R_{rs}` at NIR wavelengths from visible bands.

**Implementation**: The ``correct_atmosphere.aerosols`` module provides:

* ``angstrom_exponent(tau_1, tau_2, wavelength_1, wavelength_2)``
* ``aerosol_optical_thickness(wavelength, reference_tau, reference_wavelength, alpha)``
* ``epsilon_ratio(rho_1, rho_2)``
* ``AerosolLUT`` - Lookup tables for aerosol models
* ``AerosolModel`` - Dataclass for aerosol model parameters
* ``black_pixel_correction(...)`` - Standard aerosol retrieval
* ``should_apply_nonblack_pixel(chlorophyll)`` - Decision logic
* ``estimate_nir_rrs(...)`` - NIR Rrs estimation for turbid waters

Transmittance
=============

Both direct and diffuse transmittance are needed (Section 5.4):

Direct (beam) transmittance:

.. math::

    T = \exp\left(-\frac{\tau}{\cos\theta}\right)

Diffuse transmittance accounts for multiple scattering and depends on the
optical thickness, single-scattering albedo, and geometry.

**Implementation**: The ``correct_atmosphere.transmittance`` module provides:

* ``direct_transmittance(zenith_angle, optical_thickness)``
* ``diffuse_transmittance(zenith_angle, wavelength, aerosol_tau)``
* ``diffuse_transmittance_rayleigh(wavelength, pressure)``
* ``total_transmittance(solar_zenith, view_zenith, wavelength, aerosol_tau)``
* ``two_path_direct_transmittance(...)``

Normalized Reflectances
=======================

The normalized water-leaving radiance removes effects of solar zenith angle
and atmospheric attenuation (Section 3):

.. math::

    [L_w]_N = \left(\frac{R}{R_o}\right)^2 \frac{L_w}{\cos\theta_s \cdot t(\theta_s)}

The exact normalized reflectance includes BRDF correction (Morel et al., 2002):

.. math::

    [L_w]_N^{ex} = [L_w]_N \cdot \frac{R_o}{R(\theta'_v)} \cdot
                  \frac{f_o/Q_o}{f/Q}

where :math:`f/Q` accounts for the bidirectional reflectance of the ocean.

The remote-sensing reflectance is defined as:

.. math::

    R_{rs} = \frac{L_w}{E_d} = \frac{[\rho_w]_N}{\pi}

**Implementation**: The ``correct_atmosphere.normalization`` module provides:

* ``remote_sensing_reflectance(lw, ed)``
* ``normalized_water_leaving_radiance(lw, solar_zenith, f0, t_sun)``
* ``normalized_water_leaving_reflectance(rho_w, solar_zenith, t_sun)``
* ``earth_sun_distance_correction(day_of_year)``
* ``rrs_to_normalized_reflectance(rrs)`` - Convert :math:`R_{rs}` to :math:`[\rho_w]_N`
* ``BRDFCorrection`` - Class for f/Q correction factors
* ``snell_angle(theta_air)`` - Refraction at air-water interface

Polarization Correction
=======================

Many satellite sensors have significant polarization sensitivity (Section 11).
The measured radiance depends on the state of polarization of the incident
light, described by the Stokes vector [I, Q, U, V].

The correction is:

.. math::

    I_t = I_m - m_{12}[\cos(2\alpha)Q_R + \sin(2\alpha)U_R]
              - m_{13}[-\sin(2\alpha)Q_R + \cos(2\alpha)U_R]

where :math:`m_{12}, m_{13}` are reduced Mueller matrix elements and
:math:`\alpha` is the angle between the meridional plane and sensor reference.

**Implementation**: The ``correct_atmosphere.polarization`` module provides:

* ``StokesVector`` - Dataclass for Stokes parameters
* ``rotation_matrix(alpha)`` - 4×4 rotation matrix R(α)
* ``rotate_stokes_vector(stokes, alpha)``
* ``MuellerMatrix`` - Sensor Mueller matrix
* ``SensorPolarization.from_sensor(name)`` - Sensor-specific parameters
* ``compute_rayleigh_stokes(...)`` - Rayleigh Stokes vector at TOA
* ``polarization_correction(...)`` - Apply correction to measured radiance

Out-of-Band Correction
======================

Ocean color sensors have non-ideal spectral response functions with significant
sensitivity outside their nominal bandwidths (Section 10). This "out-of-band"
response can cause ~1% errors in TOA radiance, translating to ~10% errors in
water-leaving radiance.

The correction converts full-band measurements to equivalent values for an
idealized sensor with a perfect "top hat" response over the nominal FWHM.

**Implementation**: The ``correct_atmosphere.outofband`` module provides:

* ``gaussian_srf(wavelength, center, fwhm)`` - Gaussian SRF
* ``tophat_srf(wavelength, center, width)`` - Ideal top-hat SRF
* ``band_averaged_radiance(wavelength, radiance, srf)``
* ``OOBCorrectionLUT`` - Precomputed correction factors
* ``apply_oob_correction(rrs, sensor)`` - Apply corrections to all bands
* ``case1_rrs_spectrum(wavelength, chl)`` - Simplified bio-optical model

Supported Sensors
=================

The package includes configurations for multiple ocean color sensors:

* **SeaWiFS**: Sea-viewing Wide Field-of-view Sensor (1997-2010)
* **MODIS-Aqua**: Moderate Resolution Imaging Spectroradiometer on Aqua (2002-present)
* **MODIS-Terra**: MODIS on Terra (1999-present)
* **VIIRS-NPP**: Visible Infrared Imaging Radiometer Suite on Suomi NPP (2011-present)
* **VIIRS-NOAA20**: VIIRS on NOAA-20 (2017-present)

Each sensor has pre-defined band wavelengths, NIR reference bands for aerosol
correction, polarization sensitivity parameters, and out-of-band correction
factors.

**Implementation**: The ``correct_atmosphere.constants`` module defines:

* ``SEAWIFS_BANDS``, ``MODIS_AQUA_BANDS``, ``VIIRS_BANDS`` - Band definitions
* ``NIR_BANDS`` - NIR reference bands per sensor
* ``SENSOR_BANDS`` - Comprehensive band information
* ``get_sensor_bands(sensor)`` - Retrieve sensor configuration
* ``get_nir_bands(sensor)`` - Get NIR bands for aerosol retrieval

References
==========

Primary Reference
-----------------

* Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., & Bailey, S. (2016).
  Atmospheric Correction for Satellite Ocean Color Radiometry.
  NASA Technical Memorandum NASA/TM-2016-217551.

Rayleigh Scattering
-------------------

* Bodhaine, B.A., Wood, N.B., Dutton, E.G., & Slusser, J.R. (1999).
  On Rayleigh optical depth calculations. J. Atmos. Oceanic Technol., 16, 1854-1861.
* Wang, M. (2005). A refinement for the Rayleigh radiance computation with
  variation of the atmospheric pressure. Int. J. Remote Sens., 26, 5651-5663.

Gas Absorption
--------------

* Ahmad, Z., Franz, B.A., McClain, C.R., Kwiatkowska, E.J., Werdell, J.,
  Shettle, E.P., & Holben, B.N. (2010). New aerosol models for the retrieval
  of aerosol optical thickness and normalized water-leaving radiances from
  the SeaWiFS and MODIS sensors. Applied Optics, 49, 5545-5560.

Sun Glint
---------

* Cox, C. & Munk, W. (1954). Measurement of the roughness of the sea surface
  from photographs of the sun's glitter. J. Optical Soc. Am., 44, 838-850.
* Wang, M. & Bailey, S.W. (2001). Correction of sun glint contamination on
  the SeaWiFS ocean and atmosphere products. Applied Optics, 40, 4790-4798.

Whitecaps
---------

* Stramska, M. & Petelski, T. (2003). Observations of oceanic whitecaps in
  the north polar waters of the Atlantic. J. Geophys. Res., 108, 3086.
* Koepke, P. (1984). Effective reflectance of oceanic whitecaps.
  Applied Optics, 23, 1816-1824.

Aerosols
--------

* Gordon, H.R. & Wang, M. (1994). Retrieval of water-leaving radiance and
  aerosol optical thickness over the oceans with SeaWiFS. Applied Optics, 33, 443-452.
* Bailey, S.W., Franz, B.A., & Werdell, P.J. (2010). Estimation of near-infrared
  water-leaving reflectance for satellite ocean color data processing.
  Optics Express, 18, 7521-7527.

Normalization and BRDF
----------------------

* Morel, A., Antoine, D., & Gentili, B. (2002). Bidirectional reflectance of
  oceanic waters: accounting for Raman emission and varying particle phase
  function. Applied Optics, 41, 6289-6306.
* Morel, A. & Gentili, B. (1996). Diffuse reflectance of oceanic waters.
  III. Implication of bidirectionality for the remote-sensing problem.
  Applied Optics, 35, 4850-4862.

Polarization
------------

* Gordon, H.R., Du, T., & Zhang, T. (1997). Atmospheric correction of ocean
  color sensors: analysis of the effects of residual instrument polarization
  sensitivity. Applied Optics, 36, 6938-6948.
* Meister, G., et al. (2005). Moderate-resolution imaging spectroradiometer
  ocean color polarization correction. Applied Optics, 44, 5524-5535.

Out-of-Band Correction
----------------------

* Gordon, H.R. (1995). Remote sensing of ocean color: a methodology for
  dealing with broad spectral bands and significant out-of-band response.
  Applied Optics, 34, 8363-8374.
