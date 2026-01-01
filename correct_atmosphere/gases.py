"""
Absorption corrections for atmospheric gases.

This module implements corrections for absorption by ozone (O3) and
nitrogen dioxide (NO2) as described in Section 6.2 of Mobley et al. (2016).

Key features:

- O3 absorption using geometric air mass factor (upper atmosphere)
- NO2 absorption with multiple scattering correction (lower atmosphere)
- Band-averaged absorption cross sections for various sensors
- Temperature-dependent absorption coefficients

References
----------
.. [1] Ahmad, Z., et al. (2007). Atmospheric correction for NO2 absorption
       in retrieving water-leaving reflectances from SeaWiFS and MODIS.
       Applied Optics, 39:6504-6512.
"""

import numpy as np
from typing import Union, Optional, Dict

from oceanatmos.rayleigh import geometric_air_mass_factor
from oceanatmos.constants import (
    O3_CROSS_SECTION_SEAWIFS,
    NO2_CROSS_SECTION_SEAWIFS,
)


# Typical O3 column amounts in Dobson Units
TYPICAL_O3_DU = 350.0  # Dobson Units
MIN_O3_DU = 200.0
MAX_O3_DU = 500.0

# Conversion factor: 1 DU = 2.69e16 molecules/cm^2
DU_TO_MOLECULES_CM2 = 2.69e16

# Typical NO2 column amounts in molecules/cm^2
TYPICAL_NO2 = 1.1e16  # molecules/cm^2
LOW_NO2 = 2.8e15
HIGH_NO2 = 6.0e16


def ozone_optical_thickness(
    wavelength: Union[float, np.ndarray],
    o3_concentration: float,
    cross_section: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate ozone optical thickness.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    o3_concentration : float
        Ozone column concentration in Dobson Units (DU).
    cross_section : float, optional
        O3 absorption cross section in cm^2/molecule.
        If None, interpolated from standard values.

    Returns
    -------
    float or ndarray
        Ozone optical thickness (dimensionless).

    Notes
    -----
    Implements Equation 6.5 from Mobley et al. (2016):

    .. math::

        \\tau_{O_3}(\\lambda) = [O_3] \\cdot k_{O_3}(\\lambda)

    where [O3] is the column concentration and k_O3 is the absorption
    cross section.

    Examples
    --------
    >>> tau = ozone_optical_thickness(443.0, 350.0)
    >>> print(f"O3 optical thickness at 443 nm, 350 DU: {tau:.6f}")
    """
    # Convert DU to molecules/cm^2
    o3_molecules = o3_concentration * DU_TO_MOLECULES_CM2

    if cross_section is None:
        cross_section = interpolate_o3_cross_section(wavelength)

    tau_o3 = o3_molecules * cross_section

    return tau_o3


def interpolate_o3_cross_section(
    wavelength: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Interpolate O3 absorption cross section at given wavelength.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    float or ndarray
        O3 absorption cross section in cm^2/molecule.

    Notes
    -----
    Uses linear interpolation between tabulated band values.
    The cross sections are based on SeaWiFS band-averaged values.
    """
    # Standard wavelengths and cross sections (scaled from table values)
    wl_std = np.array([412, 443, 490, 510, 555, 670, 765, 865], dtype=float)
    k_std = np.array([0.000, 0.003, 0.021, 0.040, 0.106, 0.048, 0.007, 0.000])
    k_std = k_std * 1e-21  # Convert to cm^2/molecule

    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    cross_section = np.interp(wavelength, wl_std, k_std)

    if scalar_input:
        return float(cross_section[0])
    return cross_section


def ozone_transmittance(
    wavelength: Union[float, np.ndarray],
    o3_concentration: float,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate diffuse transmittance due to ozone absorption.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    o3_concentration : float
        Ozone column concentration in Dobson Units.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.

    Returns
    -------
    float or ndarray
        Ozone transmittance (dimensionless, 0 to 1).

    Notes
    -----
    Implements Equation 6.4 from Mobley et al. (2016):

    .. math::

        t_{O_3} = \\exp\\left[-\\tau_{O_3} \\cdot M\\right]

    where M is the geometric air mass factor. This formula is valid
    because ozone is located in the upper atmosphere where scattering
    is negligible.

    Examples
    --------
    >>> t = ozone_transmittance(443.0, 350.0, 30.0, 15.0)
    >>> print(f"O3 transmittance at 443 nm: {t:.4f}")
    """
    tau_o3 = ozone_optical_thickness(wavelength, o3_concentration)
    M = geometric_air_mass_factor(solar_zenith, view_zenith)

    t_o3 = np.exp(-tau_o3 * M)

    return t_o3


def no2_optical_thickness(
    wavelength: Union[float, np.ndarray],
    no2_concentration: float,
    cross_section: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate NO2 optical thickness.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    no2_concentration : float
        NO2 column concentration in molecules/cm^2.
    cross_section : float, optional
        NO2 absorption cross section in cm^2/molecule.
        If None, interpolated from standard values.

    Returns
    -------
    float or ndarray
        NO2 optical thickness (dimensionless).

    Examples
    --------
    >>> tau = no2_optical_thickness(412.0, 1.1e16)
    >>> print(f"NO2 optical thickness at 412 nm: {tau:.6f}")
    """
    if cross_section is None:
        cross_section = interpolate_no2_cross_section(wavelength)

    tau_no2 = no2_concentration * cross_section

    return tau_no2


def interpolate_no2_cross_section(
    wavelength: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Interpolate NO2 absorption cross section at given wavelength.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    float or ndarray
        NO2 absorption cross section in cm^2/molecule.
    """
    # Standard wavelengths and cross sections
    wl_std = np.array([412, 443, 490, 510, 555, 670, 765, 865], dtype=float)
    k_std = np.array([0.600, 0.560, 0.320, 0.210, 0.075, 0.008, 0.000, 0.000])
    k_std = k_std * 1e-19  # Convert to cm^2/molecule

    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    cross_section = np.interp(wavelength, wl_std, k_std)

    if scalar_input:
        return float(cross_section[0])
    return cross_section


def no2_correction_factor(
    wavelength: Union[float, np.ndarray],
    no2_total: float,
    no2_above_200m: float,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate NO2 correction factors for path and water-leaving radiance.

    This implements the Ahmad et al. (2007) algorithm for NO2 correction,
    which accounts for multiple scattering in the lower atmosphere.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    no2_total : float
        Total NO2 column concentration N [molecules/cm^2].
    no2_above_200m : float
        NO2 concentration above 200m, N' [molecules/cm^2].
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'path_correction': Factor for TOA path radiance correction
        - 'water_correction_solar': Factor for water-leaving radiance (solar path)
        - 'water_correction_view': Factor for water-leaving radiance (viewing path)

    Notes
    -----
    The path radiance uses the reduced concentration N' because the
    atmospheric path radiance is generated throughout the atmosphere and
    is less affected by NO2 near the surface.

    The water-leaving radiance uses the full concentration N for the
    solar path because sunlight passes through the entire atmosphere.

    See Section 6.2.2 and Equation 6.6 in Mobley et al. (2016).
    """
    alpha = interpolate_no2_cross_section(wavelength)

    theta_s = np.deg2rad(np.asarray(solar_zenith))
    theta_v = np.deg2rad(np.asarray(view_zenith))

    sec_theta_s = 1.0 / np.cos(theta_s)
    sec_theta_v = 1.0 / np.cos(theta_v)

    # Path radiance correction uses N' for both directions
    path_exponent = alpha * no2_above_200m * (sec_theta_s + sec_theta_v)
    path_correction = np.exp(path_exponent)

    # Water-leaving correction: N' for viewing path, N for solar path
    water_view_exponent = alpha * no2_above_200m * sec_theta_v
    water_solar_exponent = alpha * no2_total * sec_theta_s

    return {
        "path_correction": path_correction,
        "water_correction_view": np.exp(water_view_exponent),
        "water_correction_solar": np.exp(water_solar_exponent),
    }


def apply_no2_correction(
    rho_observed: Union[float, np.ndarray],
    rho_path: Union[float, np.ndarray],
    diffuse_transmittance: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    no2_total: float,
    no2_above_200m: float,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Apply NO2 correction to retrieve water-leaving reflectance.

    This implements the full NO2 correction algorithm from Ahmad et al. (2007)
    as described in Section 6.2.2 of Mobley et al. (2016).

    Parameters
    ----------
    rho_observed : float or array_like
        Observed TOA reflectance (uncorrected for NO2).
    rho_path : float or array_like
        Atmospheric path reflectance (Rayleigh + aerosol).
    diffuse_transmittance : float or array_like
        Diffuse transmittance t3 in viewing direction.
    wavelength : float or array_like
        Wavelength in nanometers.
    no2_total : float
        Total NO2 column concentration N [molecules/cm^2].
    no2_above_200m : float
        NO2 concentration above 200m, N' [molecules/cm^2].
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.

    Returns
    -------
    float or ndarray
        NO2-corrected water-leaving reflectance.

    Notes
    -----
    Implements Equation 6.6 from Mobley et al. (2016):

    .. math::

        \\rho'_w = \\frac{\\exp(\\alpha N' \\sec\\theta_v) 
                         \\exp(\\alpha N \\sec\\theta_s) \\Delta\\rho_{obs}}
                        {t_3 \\exp(\\alpha N' \\sec\\theta_v)}

    where :math:`\\Delta\\rho_{obs} = t_3 t_d \\rho_w` is the TOA water-leaving
    signal before NO2 correction.
    """
    corrections = no2_correction_factor(
        wavelength, no2_total, no2_above_200m, solar_zenith, view_zenith
    )

    # Correct path reflectance for NO2
    rho_path_corrected = rho_path / corrections["path_correction"]

    # Get water contribution: rho_obs - rho_path = t3 * td * rho_w
    delta_rho_obs = rho_observed - rho_path_corrected

    # Apply NO2 correction to water-leaving reflectance
    t3_corrected = diffuse_transmittance * corrections["water_correction_view"]

    rho_w_corrected = (
        corrections["water_correction_view"]
        * corrections["water_correction_solar"]
        * delta_rho_obs
        / t3_corrected
    )

    return rho_w_corrected


def gas_transmittance(
    wavelength: Union[float, np.ndarray],
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    o3_concentration: float = TYPICAL_O3_DU,
    no2_concentration: float = TYPICAL_NO2,
    include_o3: bool = True,
    include_no2: bool = True,
) -> Union[float, np.ndarray]:
    """
    Calculate total gas transmittance for O3 and NO2.

    This is a convenience function that combines O3 and NO2 transmittances
    for simple applications. For accurate NO2 correction, use the
    :func:`apply_no2_correction` function.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    o3_concentration : float, optional
        Ozone concentration in Dobson Units (default: 350 DU).
    no2_concentration : float, optional
        NO2 concentration in molecules/cm^2 (default: 1.1e16).
    include_o3 : bool, optional
        Include O3 absorption (default: True).
    include_no2 : bool, optional
        Include NO2 absorption (default: True).

    Returns
    -------
    float or ndarray
        Total gas transmittance (product of individual transmittances).

    Examples
    --------
    >>> t_gas = gas_transmittance(443.0, 30.0, 15.0)
    >>> print(f"Gas transmittance at 443 nm: {t_gas:.4f}")
    """
    transmittance = np.ones_like(np.asarray(wavelength), dtype=float)

    if include_o3:
        transmittance *= ozone_transmittance(
            wavelength, o3_concentration, solar_zenith, view_zenith
        )

    if include_no2:
        # Simple NO2 transmittance (not the full correction algorithm)
        tau_no2 = no2_optical_thickness(wavelength, no2_concentration)
        M = geometric_air_mass_factor(solar_zenith, view_zenith)
        transmittance *= np.exp(-tau_no2 * M)

    return transmittance
