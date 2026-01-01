.. _theory:

========================
Theory and Background
========================

This section provides an overview of the atmospheric correction algorithms
implemented in oceanatmos, based on NASA TM-2016-217551.

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

Rayleigh Scattering
===================

Molecular scattering by atmospheric gases follows the Rayleigh model.
The Rayleigh optical thickness at sea-level pressure is given by
Bodhaine et al. (1999):

.. math::

    \tau_R(\lambda) = 0.0021520 \frac{1.0455996 - 341.29061\lambda^{-2} - 0.90230850\lambda^2}
                                      {1.0 + 0.0027059889\lambda^{-2} - 85.968563\lambda^2}

The pressure correction follows Wang (2005).

Gas Absorption
==============

Ozone
-----

Ozone absorption transmittance:

.. math::

    t_{O_3} = \exp\left(-\tau_{O_3} M\right)

where :math:`M = 1/\cos\theta_s + 1/\cos\theta_v` is the geometric air mass
factor and :math:`\tau_{O_3} = [O_3] k_{O_3}`.

NO₂
---

NO₂ correction follows Ahmad et al. (2007), accounting for the vertical
distribution of NO₂ in the troposphere.

Sun Glint
=========

Sun glint is computed using the Cox-Munk wave slope distribution:

.. math::

    \sigma^2 = 0.00512 \cdot U

where :math:`U` is wind speed in m/s.

Pixels with normalized glint :math:`L_{GN} > 0.005` sr⁻¹ are masked.

Whitecaps
=========

Whitecap fractional coverage for undeveloped seas (Stramska & Petelski, 2003):

.. math::

    F_{wc} = 8.75 \times 10^{-5} (U - 6.33)^3

Applied for wind speeds 6.33-12 m/s.

Aerosol Correction
==================

The aerosol correction is the most challenging step. The algorithm:

1. **Black-pixel assumption**: Assume :math:`L_w = 0` at NIR wavelengths
   (valid for oligotrophic waters)
2. **Model selection**: Match measured :math:`\epsilon(\lambda_1, \lambda_2)`
   to aerosol model lookup tables
3. **Extrapolation**: Extrapolate aerosol reflectance to visible wavelengths

For waters with non-negligible NIR reflectance, the iterative non-black-pixel
algorithm (Bailey et al., 2010) is used.

Normalized Reflectances
=======================

The normalized water-leaving radiance removes effects of solar zenith angle
and atmospheric attenuation:

.. math::

    [L_w]_N = \left(\frac{R}{R_o}\right)^2 \frac{L_w}{\cos\theta_s \cdot t(\theta_s)}

The exact normalized reflectance includes BRDF correction (Morel et al., 2002):

.. math::

    [L_w]_N^{ex} = [L_w]_N \cdot \frac{R_o}{R(\theta'_v)} \cdot 
                  \frac{f_o/Q_o}{f/Q}

References
==========

* Mobley, C.D., et al. (2016). Atmospheric Correction for Satellite Ocean
  Color Radiometry. NASA/TM-2016-217551.
* Gordon, H.R. & Wang, M. (1994). Retrieval of water-leaving radiance and
  aerosol optical thickness over the oceans with SeaWiFS. Applied Optics, 33.
* Ahmad, Z., et al. (2010). New aerosol models for the retrieval of aerosol
  optical thickness. Applied Optics, 49.
* Bailey, S.W., et al. (2010). Estimation of near-infrared water-leaving
  reflectance. Optics Express, 18.
* Morel, A., et al. (2002). Bidirectional reflectance of oceanic waters.
  Applied Optics, 41.
