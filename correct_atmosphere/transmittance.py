"""
Direct and diffuse atmospheric transmittance calculations.

This module implements the atmospheric transmittance calculations described
in Chapter 4 of Mobley et al. (2016), including:

- Direct (beam) transmittance for specular reflection
- Diffuse transmittance for water-leaving radiance
- Aerosol-dependent transmittance parameterization

References
----------
.. [1] Yang, H. and Gordon, H.R. (1997). Remote sensing of ocean color:
       Assessment of the water-leaving radiance bidirectional effects on
       the atmospheric diffuse transmittance. Applied Optics, 36:7887-7897.
.. [2] Gordon, H.R. and Franz, B. (2008). Remote sensing of ocean color:
       Assessment of the water-leaving radiance bidirectional effects on
       the atmospheric diffuse transmittance for SeaWiFS and MODIS
       intercomparisons. Remote Sens. Environ., 112:2677-2685.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict

from correct_atmosphere.rayleigh import rayleigh_optical_thickness, geometric_air_mass_factor


def direct_transmittance(
    zenith_angle: Union[float, np.ndarray],
    optical_thickness: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate direct (beam) atmospheric transmittance.

    Direct transmittance applies to specular reflection where only one
    particular path connects the source and observer.

    Parameters
    ----------
    zenith_angle : float or array_like
        Zenith angle in degrees (solar or viewing).
    optical_thickness : float or array_like
        Total atmospheric optical thickness (Rayleigh + aerosol).

    Returns
    -------
    float or ndarray
        Direct transmittance (0 to 1).

    Notes
    -----
    Implements Equation 4.1 from Mobley et al. (2016):

    .. math::

        T(\\theta) = \\exp(-\\tau / \\cos\\theta)

    This is analogous to the Lambert-Beer law for radiance propagation.

    Examples
    --------
    >>> T = direct_transmittance(30.0, 0.2)
    >>> print(f"Direct transmittance: {T:.4f}")
    Direct transmittance: 0.7946
    """
    theta = np.deg2rad(np.asarray(zenith_angle))
    tau = np.asarray(optical_thickness)

    T = np.exp(-tau / np.cos(theta))

    return T


def two_path_direct_transmittance(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    optical_thickness: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate two-path direct transmittance for sun glint.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    optical_thickness : float or array_like
        Total atmospheric optical thickness.

    Returns
    -------
    float or ndarray
        Two-path direct transmittance T(theta_s) * T(theta_v).

    Notes
    -----
    For sun glint, attenuation occurs on both the downward path from sun
    to surface and the upward path from surface to sensor.

    .. math::

        T(\\theta_s) T(\\theta_v) = \\exp\\left[-\\tau M\\right]

    where M = 1/cos(theta_s) + 1/cos(theta_v) is the geometric air mass.
    """
    M = geometric_air_mass_factor(solar_zenith, view_zenith)
    tau = np.asarray(optical_thickness)

    T_two = np.exp(-tau * M)

    return T_two


def diffuse_transmittance_rayleigh(
    zenith_angle: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    pressure: float = 1013.25,
) -> Union[float, np.ndarray]:
    """
    Calculate diffuse transmittance for Rayleigh atmosphere only.

    This is the simplest approximation used for early CZCS processing.

    Parameters
    ----------
    zenith_angle : float or array_like
        Zenith angle in degrees.
    wavelength : float or array_like
        Wavelength in nanometers.
    pressure : float, optional
        Sea level pressure in hPa (default: 1013.25).

    Returns
    -------
    float or ndarray
        Diffuse transmittance.

    Notes
    -----
    For a pure Rayleigh atmosphere with no aerosols:

    .. math::

        t(\\theta) = \\exp\\left[-\\frac{1}{2} \\tau_R / \\cos\\theta\\right]

    The factor of 1/2 accounts for the difference between direct and
    diffuse transmittance in a scattering atmosphere.
    """
    tau_r = rayleigh_optical_thickness(wavelength, pressure)
    theta = np.deg2rad(np.asarray(zenith_angle))

    t = np.exp(-0.5 * tau_r / np.cos(theta))

    return t


def diffuse_transmittance_coefficients(
    aerosol_model: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Get diffuse transmittance coefficients A(theta) and B(theta) for an aerosol model.

    The diffuse transmittance is parameterized as:

    .. math::

        t^*(\\theta) = A(\\theta) \\exp[-B(\\theta) \\tau_a]

    where tau_a is the aerosol optical thickness.

    Parameters
    ----------
    aerosol_model : str
        Aerosol model identifier.

    Returns
    -------
    dict
        Dictionary with keys 'theta', 'A', 'B' containing the angle array
        and coefficient arrays.

    Notes
    -----
    These coefficients are precomputed using Monte Carlo simulations
    (Yang and Gordon, 1997, Eq. 3). In practice, they are stored in
    lookup tables for each aerosol model.

    This is a placeholder implementation returning default values.
    Actual values should be loaded from aerosol LUTs.
    """
    # Placeholder: return simple approximation coefficients
    # In practice, load from LUT files for each aerosol model
    theta = np.arange(0, 85, 5, dtype=float)
    
    # Simple approximation: A ≈ 1, B ≈ 0.5 for moderate aerosols
    A = np.ones_like(theta) * 0.98
    B = np.ones_like(theta) * 0.52

    return {"theta": theta, "A": A, "B": B}


def diffuse_transmittance(
    zenith_angle: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    aerosol_tau: Union[float, np.ndarray] = 0.0,
    aerosol_model: Optional[str] = None,
    pressure: float = 1013.25,
) -> Union[float, np.ndarray]:
    """
    Calculate diffuse atmospheric transmittance.

    Diffuse transmittance applies to water-leaving radiance where radiance
    scattered out of the beam can be replaced by radiance from neighboring
    points.

    Parameters
    ----------
    zenith_angle : float or array_like
        Zenith angle in degrees.
    wavelength : float or array_like
        Wavelength in nanometers.
    aerosol_tau : float or array_like, optional
        Aerosol optical thickness (default: 0).
    aerosol_model : str, optional
        Aerosol model for transmittance coefficients. If None, uses
        simplified calculation.
    pressure : float, optional
        Sea level pressure in hPa (default: 1013.25).

    Returns
    -------
    float or ndarray
        Diffuse transmittance (0 to 1).

    Notes
    -----
    Implements Equation 4.6 from Mobley et al. (2016):

    .. math::

        t^*(\\theta) = A(\\theta) \\exp[-B(\\theta) \\tau_a]

    where A and B are precomputed coefficients for each aerosol model.

    If no aerosol model is specified, uses a simplified approximation
    based on the Rayleigh optical thickness and aerosol optical thickness.

    The diffuse transmittance is always greater than the direct transmittance
    because scattered light can be replaced via single scattering.

    Examples
    --------
    >>> t = diffuse_transmittance(30.0, 550.0, aerosol_tau=0.1)
    >>> print(f"Diffuse transmittance: {t:.4f}")
    """
    theta = np.asarray(zenith_angle)
    tau_a = np.asarray(aerosol_tau)
    tau_r = rayleigh_optical_thickness(wavelength, pressure)

    if aerosol_model is not None:
        # Use precomputed coefficients
        coeffs = diffuse_transmittance_coefficients(aerosol_model)
        A = np.interp(theta, coeffs["theta"], coeffs["A"])
        B = np.interp(theta, coeffs["theta"], coeffs["B"])
        t = A * np.exp(-B * tau_a)
    else:
        # Simplified approximation
        # Combine Rayleigh and aerosol contributions
        cos_theta = np.cos(np.deg2rad(theta))

        # Rayleigh contribution (factor 0.5 for diffuse)
        t_rayleigh = np.exp(-0.5 * tau_r / cos_theta)

        # Aerosol contribution (factor ~0.5-0.6 depending on aerosol type)
        t_aerosol = np.exp(-0.52 * tau_a / cos_theta)

        t = t_rayleigh * t_aerosol

    return t


def brdf_transmittance_correction(
    zenith_angle: Union[float, np.ndarray],
    azimuth_angle: Union[float, np.ndarray],
    solar_zenith: Union[float, np.ndarray],
    aerosol_tau: float,
    chlorophyll: float,
) -> Union[float, np.ndarray]:
    """
    Calculate BRDF correction to diffuse transmittance.

    This correction accounts for the non-isotropic angular distribution
    of water-leaving radiance.

    Parameters
    ----------
    zenith_angle : float or array_like
        Viewing zenith angle in degrees.
    azimuth_angle : float or array_like
        Viewing azimuth angle relative to sun in degrees.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    aerosol_tau : float
        Aerosol optical thickness.
    chlorophyll : float
        Chlorophyll concentration in mg/m^3.

    Returns
    -------
    float or ndarray
        BRDF correction factor delta(theta_v, phi_v).

    Notes
    -----
    From Gordon and Franz (2008), this correction improves the diffuse
    transmittance for off-nadir viewing directions. The correction is:

    .. math::

        t(\\theta_v, \\phi_v) = t^*(\\theta_v) [1 + \\delta(\\theta_v, \\phi_v)]

    For viewing angles < 60 deg, the correction is typically small (< 1%).

    This is a placeholder implementation. Actual values require the
    Lu(theta', phi') angular distribution from f/Q lookup tables.
    """
    # Placeholder: return zero correction
    # In practice, this requires f/Q lookup tables parameterized by Chl
    theta_v = np.asarray(zenith_angle)
    
    # Simple approximation: correction increases with viewing angle
    # and is larger near the principal plane (phi = 0 or 180)
    phi = np.deg2rad(np.asarray(azimuth_angle))
    theta_v_rad = np.deg2rad(theta_v)

    # Very rough approximation
    delta = 0.01 * (theta_v / 60.0) ** 2 * np.abs(np.cos(phi))

    return delta


def total_transmittance(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    aerosol_tau: Union[float, np.ndarray] = 0.0,
    direction: str = "both",
    pressure: float = 1013.25,
) -> Union[float, np.ndarray]:
    """
    Calculate total diffuse transmittance.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    wavelength : float or array_like
        Wavelength in nanometers.
    aerosol_tau : float or array_like, optional
        Aerosol optical thickness (default: 0).
    direction : str, optional
        Which path(s) to include:
        - 'solar': Only sun-to-surface path t(theta_s)
        - 'view': Only surface-to-sensor path t(theta_v)
        - 'both': Both paths t(theta_s) * t(theta_v) (default)
    pressure : float, optional
        Sea level pressure in hPa.

    Returns
    -------
    float or ndarray
        Total diffuse transmittance.

    Examples
    --------
    >>> t = total_transmittance(30.0, 15.0, 550.0, aerosol_tau=0.1)
    >>> print(f"Total transmittance: {t:.4f}")
    """
    if direction == "solar":
        return diffuse_transmittance(solar_zenith, wavelength, aerosol_tau, pressure=pressure)
    elif direction == "view":
        return diffuse_transmittance(view_zenith, wavelength, aerosol_tau, pressure=pressure)
    elif direction == "both":
        t_s = diffuse_transmittance(solar_zenith, wavelength, aerosol_tau, pressure=pressure)
        t_v = diffuse_transmittance(view_zenith, wavelength, aerosol_tau, pressure=pressure)
        return t_s * t_v
    else:
        raise ValueError(f"direction must be 'solar', 'view', or 'both', got '{direction}'")


def gaseous_transmittance(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    o3_concentration: float = 350.0,
    no2_concentration: float = 1.1e16,
) -> Union[float, np.ndarray]:
    """
    Calculate gaseous transmittance (O3 and NO2).

    This is a convenience wrapper around the gas absorption functions.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.
    wavelength : float or array_like
        Wavelength in nanometers.
    o3_concentration : float, optional
        Ozone concentration in Dobson Units (default: 350).
    no2_concentration : float, optional
        NO2 concentration in molecules/cm^2 (default: 1.1e16).

    Returns
    -------
    float or ndarray
        Gaseous transmittance t_gv * t_gs.

    Notes
    -----
    See the :mod:`correct_atmosphere.gases` module for detailed gas absorption
    calculations.
    """
    from correct_atmosphere.gases import gas_transmittance

    return gas_transmittance(
        wavelength,
        solar_zenith,
        view_zenith,
        o3_concentration,
        no2_concentration,
    )
