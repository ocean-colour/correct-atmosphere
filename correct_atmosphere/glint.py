"""
Sun glint correction algorithms.

This module implements the sun glint correction described in Chapter 7
of Mobley et al. (2016), including:

- Normalized sun glint calculation using Cox-Munk wave slope distribution
- Two-path transmittance for glint radiance
- Glint masking based on threshold

References
----------
.. [1] Wang, M. and Bailey, S.W. (2001). Correction of sun glint contamination
       of the SeaWiFS ocean and atmosphere products. Applied Optics,
       40:4790-4798.
.. [2] Cox, C. and Munk, W. (1954). Measurement of the roughness of the sea
       surface from photographs of the sun's glitter. J. Opt. Soc. Amer.,
       44:838-850.
"""

import numpy as np
from typing import Union, Tuple, Optional

from correct_atmosphere.constants import GLINT_THRESHOLD
from correct_atmosphere.rayleigh import rayleigh_optical_thickness


def cox_munk_slope_variance(wind_speed: float) -> float:
    """
    Calculate mean square sea surface slope from wind speed.

    Uses the Cox-Munk (1954) empirical relationship.

    Parameters
    ----------
    wind_speed : float
        Wind speed at 10m height in m/s.

    Returns
    -------
    float
        Mean square slope (dimensionless).

    Notes
    -----
    The relationship is:

    .. math::

        \\sigma^2 = 0.00512 \\cdot U

    where U is the wind speed in m/s. The LUT wind speeds (0, 1.9, 4.2,
    7.5, 11.7, ...) correspond to convenient spacing in mean square slopes.

    Examples
    --------
    >>> mss = cox_munk_slope_variance(10.0)
    >>> print(f"Mean square slope at 10 m/s: {mss:.4f}")
    Mean square slope at 10 m/s: 0.0512
    """
    return 0.00512 * wind_speed


def wave_facet_probability(
    slope_x: Union[float, np.ndarray],
    slope_y: Union[float, np.ndarray],
    wind_speed: float,
) -> Union[float, np.ndarray]:
    """
    Calculate probability of wave facet slope using isotropic Cox-Munk.

    Parameters
    ----------
    slope_x : float or array_like
        Cross-wind slope component.
    slope_y : float or array_like
        Along-wind slope component.
    wind_speed : float
        Wind speed at 10m in m/s.

    Returns
    -------
    float or ndarray
        Probability density of the slope.

    Notes
    -----
    Uses the azimuthally isotropic form of the Cox-Munk distribution:

    .. math::

        p(z_x, z_y) = \\frac{1}{\\pi \\sigma^2} 
                      \\exp\\left(-\\frac{z_x^2 + z_y^2}{\\sigma^2}\\right)

    where :math:`\\sigma^2` is the mean square slope.
    """
    sigma_sq = cox_munk_slope_variance(wind_speed)

    if sigma_sq == 0:
        # Level surface: delta function at zero slope
        slope_sq = np.asarray(slope_x) ** 2 + np.asarray(slope_y) ** 2
        return np.where(slope_sq == 0, np.inf, 0.0)

    slope_sq = np.asarray(slope_x) ** 2 + np.asarray(slope_y) ** 2
    p = np.exp(-slope_sq / sigma_sq) / (np.pi * sigma_sq)

    return p


def calculate_specular_slope(
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
) -> Tuple[float, float, float]:
    """
    Calculate wave facet slope required for specular reflection.

    Parameters
    ----------
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle (view - sun) in degrees.

    Returns
    -------
    tuple of float
        (slope_x, slope_y, cos_beta) where slope_x and slope_y are the
        required facet slopes and cos_beta is the cosine of the angle
        between surface normal and facet normal.

    Notes
    -----
    For specular reflection, the wave facet normal must bisect the angle
    between the incident and reflected rays.
    """
    theta_s = np.deg2rad(solar_zenith)
    theta_v = np.deg2rad(view_zenith)
    phi = np.deg2rad(relative_azimuth)

    # Unit vectors for sun and view directions
    # (pointing toward sun and toward sensor)
    sun_x = np.sin(theta_s)
    sun_y = 0.0
    sun_z = np.cos(theta_s)

    view_x = np.sin(theta_v) * np.cos(phi)
    view_y = np.sin(theta_v) * np.sin(phi)
    view_z = np.cos(theta_v)

    # Facet normal bisects the angle
    norm_x = sun_x + view_x
    norm_y = sun_y + view_y
    norm_z = sun_z + view_z

    # Normalize
    norm_mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x /= norm_mag
    norm_y /= norm_mag
    norm_z /= norm_mag

    # Slopes
    if norm_z > 0:
        slope_x = -norm_x / norm_z
        slope_y = -norm_y / norm_z
    else:
        slope_x = np.inf
        slope_y = np.inf

    # cos(beta) = facet normal · vertical
    cos_beta = norm_z

    return slope_x, slope_y, cos_beta


def normalized_sun_glint(
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
    wind_speed: float,
) -> float:
    """
    Calculate normalized sun glint L_GN.

    Parameters
    ----------
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle (view - sun) in degrees.
    wind_speed : float
        Wind speed at 10m in m/s.

    Returns
    -------
    float
        Normalized sun glint L_GN in sr^-1.

    Notes
    -----
    L_GN is computed using the Cox-Munk wave slope distribution and an
    incident irradiance of F0 = 1 W/(m^2 nm). It represents the angular
    distribution of reflected radiance but has units of 1/steradian.

    L_GN is independent of wavelength. The actual glint radiance is:

    .. math::

        L_g(\\lambda) = F_0(\\lambda) T(\\theta_s, \\lambda) L_{GN}

    Examples
    --------
    >>> lgn = normalized_sun_glint(30.0, 15.0, 90.0, 10.0)
    >>> print(f"Normalized sun glint: {lgn:.6f} sr^-1")
    """
    theta_s = np.deg2rad(solar_zenith)
    theta_v = np.deg2rad(view_zenith)

    # Get required facet slope for specular reflection
    slope_x, slope_y, cos_beta = calculate_specular_slope(
        solar_zenith, view_zenith, relative_azimuth
    )

    if np.isinf(slope_x):
        return 0.0

    # Probability of this slope
    p = wave_facet_probability(slope_x, slope_y, wind_speed)

    # Fresnel reflectance at the incidence angle
    # Incidence angle on facet = angle between sun direction and facet normal
    cos_incidence = np.cos(theta_s) * cos_beta + np.sin(theta_s) * np.sqrt(
        1 - cos_beta**2
    )
    cos_incidence = min(1.0, max(-1.0, cos_incidence))

    # Simplified Fresnel reflectance for water (n ≈ 1.34)
    n = 1.34
    sin_incidence = np.sqrt(1 - cos_incidence**2)
    sin_refract = sin_incidence / n

    if sin_refract > 1:
        # Total internal reflection (shouldn't happen for air-to-water)
        rho_fresnel = 1.0
    else:
        cos_refract = np.sqrt(1 - sin_refract**2)
        rs = ((n * cos_incidence - cos_refract) / (n * cos_incidence + cos_refract)) ** 2
        rp = ((cos_incidence - n * cos_refract) / (cos_incidence + n * cos_refract)) ** 2
        rho_fresnel = 0.5 * (rs + rp)

    # Normalized sun glint
    # L_GN = rho * p / (4 * cos(theta_s) * cos(theta_v) * cos^4(beta))
    cos_beta_4 = cos_beta**4

    if cos_beta_4 > 0 and np.cos(theta_s) > 0 and np.cos(theta_v) > 0:
        L_GN = rho_fresnel * p / (4.0 * np.cos(theta_s) * np.cos(theta_v) * cos_beta_4)
    else:
        L_GN = 0.0

    return L_GN


def glint_mask(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    relative_azimuth: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    threshold: float = GLINT_THRESHOLD,
) -> Union[bool, np.ndarray]:
    """
    Determine if pixels should be masked due to excessive sun glint.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    relative_azimuth : float or array_like
        Relative azimuth angle in degrees.
    wind_speed : float or array_like
        Wind speed at 10m in m/s.
    threshold : float, optional
        L_GN threshold above which pixels are masked (default: 0.005 sr^-1).

    Returns
    -------
    bool or ndarray
        True if pixel should be masked (too much glint), False otherwise.

    Examples
    --------
    >>> mask = glint_mask(30.0, 15.0, 90.0, 10.0)
    >>> print(f"Should mask: {mask}")
    """
    solar_zenith = np.asarray(solar_zenith)
    view_zenith = np.asarray(view_zenith)
    relative_azimuth = np.asarray(relative_azimuth)
    wind_speed = np.asarray(wind_speed)

    scalar_input = solar_zenith.ndim == 0

    if scalar_input:
        lgn = normalized_sun_glint(
            float(solar_zenith),
            float(view_zenith),
            float(relative_azimuth),
            float(wind_speed),
        )
        return lgn > threshold
    else:
        # Vectorized version
        mask = np.zeros(solar_zenith.shape, dtype=bool)
        for idx in np.ndindex(solar_zenith.shape):
            lgn = normalized_sun_glint(
                solar_zenith[idx],
                view_zenith[idx],
                relative_azimuth[idx],
                wind_speed[idx] if wind_speed.shape else float(wind_speed),
            )
            mask[idx] = lgn > threshold
        return mask


def direct_transmittance(
    zenith_angle: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    rayleigh_tau: Optional[Union[float, np.ndarray]] = None,
    aerosol_tau: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate direct (beam) atmospheric transmittance.

    Parameters
    ----------
    zenith_angle : float or array_like
        Zenith angle in degrees (solar or viewing).
    wavelength : float or array_like
        Wavelength in nanometers.
    rayleigh_tau : float or array_like, optional
        Rayleigh optical thickness. If None, computed from wavelength.
    aerosol_tau : float, optional
        Aerosol optical thickness (default: 0).

    Returns
    -------
    float or ndarray
        Direct transmittance.

    Notes
    -----
    Direct transmittance is used for specular reflection (sun glint).
    Implements Equation 4.1 from Mobley et al. (2016):

    .. math::

        T(\\theta) = \\exp(-\\tau / \\cos\\theta)

    where :math:`\\tau = \\tau_R + \\tau_a` is the total optical thickness.
    """
    if rayleigh_tau is None:
        rayleigh_tau = rayleigh_optical_thickness(wavelength)

    tau_total = np.asarray(rayleigh_tau) + aerosol_tau
    theta = np.deg2rad(np.asarray(zenith_angle))

    T = np.exp(-tau_total / np.cos(theta))

    return T


def two_path_transmittance(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    rayleigh_tau: Optional[Union[float, np.ndarray]] = None,
    aerosol_tau: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate two-path direct transmittance for sun glint.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    wavelength : float or array_like
        Wavelength in nanometers.
    rayleigh_tau : float or array_like, optional
        Rayleigh optical thickness.
    aerosol_tau : float, optional
        Aerosol optical thickness.

    Returns
    -------
    float or ndarray
        Two-path transmittance T(theta_s) * T(theta_v).

    Notes
    -----
    Implements Equation 7.1 from Mobley et al. (2016):

    .. math::

        T(\\theta_s) T(\\theta_v) = \\exp\\left[-\\tau 
        \\left(\\frac{1}{\\cos\\theta_s} + \\frac{1}{\\cos\\theta_v}\\right)\\right]
    """
    if rayleigh_tau is None:
        rayleigh_tau = rayleigh_optical_thickness(wavelength)

    tau_total = np.asarray(rayleigh_tau) + aerosol_tau
    theta_s = np.deg2rad(np.asarray(solar_zenith))
    theta_v = np.deg2rad(np.asarray(view_zenith))

    T_two_path = np.exp(-tau_total * (1.0 / np.cos(theta_s) + 1.0 / np.cos(theta_v)))

    return T_two_path


def sun_glint_reflectance(
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
    wind_speed: float,
    wavelength: Union[float, np.ndarray],
    aerosol_tau: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate sun glint reflectance at TOA.

    This combines the normalized sun glint with the two-path transmittance
    to give the glint contribution to the TOA signal.

    Parameters
    ----------
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle in degrees.
    wind_speed : float
        Wind speed at 10m in m/s.
    wavelength : float or array_like
        Wavelength in nanometers.
    aerosol_tau : float, optional
        Aerosol optical thickness.

    Returns
    -------
    float or ndarray
        Sun glint reflectance at TOA.

    Notes
    -----
    The glint reflectance to be subtracted from the TOA reflectance is:

    .. math::

        T(\\theta_s) T(\\theta_v) L_{GN} \\cdot \\pi / F_0

    Since L_GN is already normalized to F0 = 1, the factor pi converts
    the radiance-based L_GN to a reflectance.
    """
    L_GN = normalized_sun_glint(solar_zenith, view_zenith, relative_azimuth, wind_speed)

    T_two = two_path_transmittance(
        solar_zenith, view_zenith, wavelength, aerosol_tau=aerosol_tau
    )

    # Convert to reflectance (L_GN is per unit F0, multiply by pi)
    rho_glint = np.pi * L_GN * T_two

    return rho_glint
