"""
Whitecap and foam reflectance calculations.

This module implements the whitecap correction described in Chapter 8
of Mobley et al. (2016), including:

- Fractional whitecap coverage as a function of wind speed
- Whitecap reflectance with spectral dependence
- Normalized whitecap reflectance

References
----------
.. [1] Gordon, H.R. and Wang, M. (1994). Influence of oceanic whitecaps on
       atmospheric correction of ocean-color sensors. Applied Optics,
       33:7754-7763.
.. [2] Koepke, P. (1984). Effective reflectance of oceanic whitecaps.
       Applied Optics, 23:1816-1824.
.. [3] Stramska, M. and Petelski, T. (2003). Observations of oceanic
       whitecaps in the north polar waters of the Atlantic. J. Geophys. Res.,
       108, doi:10.1029/2002JC001321.
.. [4] Frouin, R., et al. (1996). Spectral reflectance of sea foam in the
       visible and near infrared. J. Geophys. Res., 101:14361-14371.
"""

import numpy as np
from typing import Union

from oceanatmos.constants import (
    WHITECAP_REFLECTANCE_EFFECTIVE,
    WHITECAP_SPECTRAL_FACTOR,
    WHITECAP_MIN_WIND_SPEED,
    WHITECAP_MAX_WIND_SPEED,
)


def whitecap_fraction_developed(wind_speed: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractional whitecap coverage for developed seas.

    Parameters
    ----------
    wind_speed : float or array_like
        Wind speed at 10m in m/s.

    Returns
    -------
    float or ndarray
        Fractional whitecap coverage (0 to 1).

    Notes
    -----
    Implements Equation 8.2 from Mobley et al. (2016), from
    Stramska and Petelski (2003):

    .. math::

        F_{wc} = 5.0 \\times 10^{-5} (U_{10} - 4.47)^3

    for U10 > 4.47 m/s.

    Examples
    --------
    >>> frac = whitecap_fraction_developed(15.0)
    >>> print(f"Whitecap fraction at 15 m/s: {frac:.4f}")
    """
    wind_speed = np.asarray(wind_speed)
    threshold = 4.47

    frac = np.where(
        wind_speed > threshold,
        5.0e-5 * (wind_speed - threshold) ** 3,
        0.0,
    )

    return frac


def whitecap_fraction_undeveloped(wind_speed: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractional whitecap coverage for undeveloped seas.

    This is the default model used for remote sensing because if seas
    are well developed, it is likely stormy and cloudy.

    Parameters
    ----------
    wind_speed : float or array_like
        Wind speed at 10m in m/s.

    Returns
    -------
    float or ndarray
        Fractional whitecap coverage (0 to 1).

    Notes
    -----
    Implements Equation 8.3 from Mobley et al. (2016), from
    Stramska and Petelski (2003):

    .. math::

        F_{wc} = 8.75 \\times 10^{-5} (U_{10} - 6.33)^3

    for U10 > 6.33 m/s.

    Examples
    --------
    >>> frac = whitecap_fraction_undeveloped(10.0)
    >>> print(f"Whitecap fraction at 10 m/s: {frac:.4f}")
    """
    wind_speed = np.asarray(wind_speed)
    threshold = WHITECAP_MIN_WIND_SPEED  # 6.33 m/s

    frac = np.where(
        wind_speed > threshold,
        8.75e-5 * (wind_speed - threshold) ** 3,
        0.0,
    )

    return frac


def whitecap_fraction(
    wind_speed: Union[float, np.ndarray],
    developed_sea: bool = False,
) -> Union[float, np.ndarray]:
    """
    Calculate fractional whitecap coverage.

    Parameters
    ----------
    wind_speed : float or array_like
        Wind speed at 10m in m/s.
    developed_sea : bool, optional
        If True, use formula for developed seas (default: False).

    Returns
    -------
    float or ndarray
        Fractional whitecap coverage (0 to 1).

    Examples
    --------
    >>> frac = whitecap_fraction(10.0)
    >>> print(f"Whitecap fraction at 10 m/s: {frac:.4f}")
    Whitecap fraction at 10 m/s: 0.0179
    """
    if developed_sea:
        return whitecap_fraction_developed(wind_speed)
    else:
        return whitecap_fraction_undeveloped(wind_speed)


def whitecap_spectral_factor(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate wavelength-dependent whitecap reflectance factor.

    Whitecaps are less reflective at red and NIR wavelengths than at
    blue wavelengths due to water absorption in the foam.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    float or ndarray
        Spectral factor a_wc (0 to 1, relative to blue wavelengths).

    Notes
    -----
    Values are from Frouin et al. (1996). The factor is 1.0 for
    wavelengths <= 555 nm and decreases at longer wavelengths.
    """
    # Standard values from Frouin et al. (1996)
    wl_std = np.array([412, 443, 490, 510, 555, 670, 765, 865], dtype=float)
    a_std = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.889, 0.760, 0.645])

    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    # Interpolate
    a_wc = np.interp(wavelength, wl_std, a_std)

    # Clamp to [0, 1]
    a_wc = np.clip(a_wc, 0.0, 1.0)

    if scalar_input:
        return float(a_wc[0])
    return a_wc


def whitecap_reflectance(
    wind_speed: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    developed_sea: bool = False,
) -> Union[float, np.ndarray]:
    """
    Calculate normalized whitecap reflectance [rho_wc]_N.

    Parameters
    ----------
    wind_speed : float or array_like
        Wind speed at 10m in m/s.
    wavelength : float or array_like
        Wavelength in nanometers.
    developed_sea : bool, optional
        If True, use formula for developed seas (default: False).

    Returns
    -------
    float or ndarray
        Normalized whitecap reflectance (dimensionless).

    Notes
    -----
    Implements Equation 8.4 from Mobley et al. (2016):

    .. math::

        [\\rho_{wc}]_N(\\lambda) = a_{wc}(\\lambda) \\times 0.22 \\times F_{wc}

    where:

    - :math:`a_{wc}` is the spectral factor from Frouin et al. (1996)
    - 0.22 is the effective whitecap reflectance from Koepke (1984)
    - :math:`F_{wc}` is the fractional whitecap coverage

    A whitecap correction is applied only for wind speeds in the range
    6.33 to 12 m/s.

    Examples
    --------
    >>> rho = whitecap_reflectance(10.0, 550.0)
    >>> print(f"Whitecap reflectance at 10 m/s, 550 nm: {rho:.6f}")
    
    >>> wavelengths = np.array([412, 555, 670, 865])
    >>> rho = whitecap_reflectance(10.0, wavelengths)
    """
    wind_speed = np.asarray(wind_speed)
    wavelength = np.asarray(wavelength)

    # Get fractional coverage
    frac = whitecap_fraction(wind_speed, developed_sea)

    # Get spectral factor
    a_wc = whitecap_spectral_factor(wavelength)

    # Effective reflectance
    rho_eff = WHITECAP_REFLECTANCE_EFFECTIVE  # 0.22

    # Normalized whitecap reflectance
    rho_wc = a_wc * rho_eff * frac

    # Apply wind speed limits
    # Only apply correction for WHITECAP_MIN_WIND_SPEED <= U <= WHITECAP_MAX_WIND_SPEED
    wind_mask = (wind_speed >= WHITECAP_MIN_WIND_SPEED) & (
        wind_speed <= WHITECAP_MAX_WIND_SPEED
    )

    if np.ndim(wind_mask) == 0:
        if not wind_mask:
            if np.ndim(rho_wc) == 0:
                return 0.0
            return np.zeros_like(rho_wc)
    else:
        rho_wc = np.where(wind_mask[..., np.newaxis] if np.ndim(wavelength) > 0 else wind_mask, 
                         rho_wc, 0.0)

    return rho_wc


def whitecap_radiance(
    wind_speed: float,
    wavelength: Union[float, np.ndarray],
    solar_irradiance: Union[float, np.ndarray],
    solar_zenith: float,
    diffuse_transmittance: Union[float, np.ndarray],
    developed_sea: bool = False,
) -> Union[float, np.ndarray]:
    """
    Calculate whitecap radiance at sea surface.

    Parameters
    ----------
    wind_speed : float
        Wind speed at 10m in m/s.
    wavelength : float or array_like
        Wavelength in nanometers.
    solar_irradiance : float or array_like
        Extraterrestrial solar irradiance F0 [W/(m^2 nm)].
    solar_zenith : float
        Solar zenith angle in degrees.
    diffuse_transmittance : float or array_like
        Diffuse atmospheric transmittance in sun direction.
    developed_sea : bool, optional
        If True, use formula for developed seas (default: False).

    Returns
    -------
    float or ndarray
        Whitecap radiance at sea surface [W/(m^2 sr nm)].

    Notes
    -----
    Based on Equation 8.1 from Mobley et al. (2016). The whitecap
    radiance is:

    .. math::

        L_{wc} = \\frac{[\\rho_{wc}]_N F_0 \\cos\\theta_s t(\\theta_s)}{\\pi}

    assuming Lambertian reflection from whitecaps.
    """
    rho_wc = whitecap_reflectance(wind_speed, wavelength, developed_sea)

    cos_theta_s = np.cos(np.deg2rad(solar_zenith))

    # Incident irradiance at surface
    E_surface = solar_irradiance * cos_theta_s * diffuse_transmittance

    # Lambertian radiance
    L_wc = rho_wc * E_surface / np.pi

    return L_wc


def whitecap_toa_contribution(
    wind_speed: float,
    wavelength: Union[float, np.ndarray],
    diffuse_transmittance_sun: Union[float, np.ndarray],
    diffuse_transmittance_view: Union[float, np.ndarray],
    developed_sea: bool = False,
) -> Union[float, np.ndarray]:
    """
    Calculate whitecap contribution to TOA reflectance.

    Parameters
    ----------
    wind_speed : float
        Wind speed at 10m in m/s.
    wavelength : float or array_like
        Wavelength in nanometers.
    diffuse_transmittance_sun : float or array_like
        Diffuse transmittance in sun direction t(theta_s).
    diffuse_transmittance_view : float or array_like
        Diffuse transmittance in view direction t(theta_v).
    developed_sea : bool, optional
        If True, use formula for developed seas (default: False).

    Returns
    -------
    float or ndarray
        Whitecap contribution to TOA reflectance.

    Notes
    -----
    From Equation 3.13 of Mobley et al. (2016), the whitecap term at TOA is:

    .. math::

        t(\\theta_v) \\rho_{wc} = [\\rho_{wc}]_N t(\\theta_s) t(\\theta_v)
    """
    rho_wc = whitecap_reflectance(wind_speed, wavelength, developed_sea)

    # Two-path diffuse transmittance
    rho_wc_toa = rho_wc * diffuse_transmittance_sun * diffuse_transmittance_view

    return rho_wc_toa
