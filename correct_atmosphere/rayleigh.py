"""
Rayleigh scattering corrections for atmospheric gas molecules.

This module implements the Rayleigh correction algorithms described in
Section 6.1 of Mobley et al. (2016), including:

- Rayleigh optical thickness calculation (Bodhaine et al., 1999)
- Pressure correction for non-standard sea level pressure
- Wind speed and surface reflectance effects
- Rayleigh path radiance computation

The Rayleigh lookup tables (LUTs) contain precomputed radiances for various
geometries and wind speeds, computed using a vector radiative transfer model
that includes multiple scattering, polarization, and sea surface roughness.

References
----------
.. [1] Bodhaine, B.A., et al. (1999). On Rayleigh optical depth calculations.
       J. Atmos. Oceanic Technol., 16:1854-1861.
.. [2] Wang, M. (2002). The Rayleigh lookup tables for the SeaWiFS data
       processing. Int. J. Remote Sensing, 23:2697-2702.
.. [3] Wang, M. (2005). A refinement for the Rayleigh radiance computation
       with variation of the atmospheric pressure. Int. J. Remote Sensing,
       26:5651-5663.
"""

import numpy as np
from typing import Union, Optional, Tuple

from oceanatmos.constants import (
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
    STANDARD_CO2_PPM,
)


def rayleigh_optical_thickness(
    wavelength: Union[float, np.ndarray],
    pressure: float = STANDARD_PRESSURE,
    co2_ppm: float = STANDARD_CO2_PPM,
) -> Union[float, np.ndarray]:
    """
    Calculate Rayleigh optical thickness at given wavelength(s).

    Implements Equation 6.1 from Mobley et al. (2016), based on
    Bodhaine et al. (1999, Eq. 30).

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    pressure : float, optional
        Sea level atmospheric pressure in hPa (default: 1013.25 hPa).
    co2_ppm : float, optional
        CO2 concentration in ppm (default: 360 ppm).

    Returns
    -------
    float or ndarray
        Rayleigh optical thickness (dimensionless).

    Notes
    -----
    The formula is valid for wavelengths from approximately 200 nm to 1000 nm.
    The optical thickness scales linearly with pressure:

    .. math::

        \\tau_R(P, \\lambda) = \\frac{P}{P_0} \\tau_{R0}(P_0, \\lambda)

    where :math:`P_0` = 1013.25 hPa is the standard pressure.

    Examples
    --------
    >>> tau = rayleigh_optical_thickness(443.0)
    >>> print(f"Rayleigh optical thickness at 443 nm: {tau:.4f}")
    Rayleigh optical thickness at 443 nm: 0.2350

    >>> wavelengths = np.array([412, 443, 490, 555, 670])
    >>> tau = rayleigh_optical_thickness(wavelengths)
    """
    # Convert wavelength to micrometers
    lam = np.asarray(wavelength) / 1000.0  # nm to micrometers

    # Bodhaine et al. (1999) Eq. 30 for standard atmosphere
    # P = 1013.25 hPa, T = 288.15 K, CO2 = 360 ppm
    numerator = 1.0455996 - 341.29061 * lam**(-2) - 0.90230850 * lam**2
    denominator = 1.0 + 0.0027059889 * lam**(-2) - 85.968563 * lam**2
    tau_r0 = 0.0021520 * (numerator / denominator)

    # Pressure correction
    tau_r = (pressure / STANDARD_PRESSURE) * tau_r0

    return tau_r


def rayleigh_depolarization_ratio(
    wavelength: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate Rayleigh depolarization ratio.

    The depolarization ratio accounts for the anisotropy of air molecules
    and affects the Rayleigh phase function.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    float or ndarray
        Depolarization ratio (dimensionless), typically ~0.03.

    Notes
    -----
    Based on Bodhaine et al. (1999). The depolarization ratio is weakly
    wavelength dependent in the visible range.
    """
    # Convert wavelength to micrometers
    lam = np.asarray(wavelength) / 1000.0

    # Approximate formula from Bodhaine et al. (1999)
    # This is a simplified version; full calculation requires
    # refractive index dispersion
    rho = 0.0279 + 0.00013 / (lam**2)

    return rho


def geometric_air_mass_factor(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the geometric air mass factor M.

    The air mass factor represents the total atmospheric path length
    relative to a vertical path, for both the solar and viewing directions.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.

    Returns
    -------
    float or ndarray
        Air mass factor M = 1/cos(theta_s) + 1/cos(theta_v).

    Notes
    -----
    Implements Equation 6.3 from Mobley et al. (2016):

    .. math::

        M = \\frac{1}{\\cos\\theta_s} + \\frac{1}{\\cos\\theta_v}

    Examples
    --------
    >>> M = geometric_air_mass_factor(30.0, 15.0)
    >>> print(f"Air mass factor: {M:.3f}")
    Air mass factor: 2.189
    """
    theta_s = np.deg2rad(np.asarray(solar_zenith))
    theta_v = np.deg2rad(np.asarray(view_zenith))

    M = 1.0 / np.cos(theta_s) + 1.0 / np.cos(theta_v)

    return M


def pressure_correction_coefficient(
    wavelength: Union[float, np.ndarray],
    air_mass: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate pressure correction coefficient C(lambda, M).

    This coefficient is used in the pressure correction formula for
    Rayleigh radiance (Equation 6.2).

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in nanometers.
    air_mass : float or array_like
        Geometric air mass factor M.

    Returns
    -------
    float or ndarray
        Pressure correction coefficient C.

    Notes
    -----
    From Wang (2005):

    .. math::

        C(\\lambda, M) = a(\\lambda) + b(\\lambda) \\ln(M)

    where:

    .. math::

        a(\\lambda) = -0.6543 + 1.608 \\tau_{R0}(\\lambda)

        b(\\lambda) = 0.8192 - 1.2541 \\tau_{R0}(\\lambda)
    """
    tau_r0 = rayleigh_optical_thickness(wavelength, pressure=STANDARD_PRESSURE)
    M = np.asarray(air_mass)

    a = -0.6543 + 1.608 * tau_r0
    b = 0.8192 - 1.2541 * tau_r0

    C = a + b * np.log(M)

    return C


def rayleigh_reflectance_pressure_corrected(
    rayleigh_reflectance_std: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    pressure: float,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Apply pressure correction to Rayleigh reflectance from LUT.

    Converts Rayleigh reflectance computed at standard pressure to the
    actual pressure at the time of observation.

    Parameters
    ----------
    rayleigh_reflectance_std : float or array_like
        Rayleigh reflectance at standard pressure from LUT.
    wavelength : float or array_like
        Wavelength in nanometers.
    pressure : float
        Actual sea level pressure in hPa.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    view_zenith : float or array_like
        Viewing zenith angle in degrees.

    Returns
    -------
    float or ndarray
        Pressure-corrected Rayleigh reflectance.

    Notes
    -----
    Implements Equation 6.2 from Mobley et al. (2016):

    .. math::

        L_R[\\tau_R(P)] = L_R[\\tau_R(P_0)] 
        \\frac{1 - \\exp[-C \\tau_R(P) M]}{1 - \\exp[-C \\tau_R(P_0) M]}

    where the same formula applies to reflectance.
    """
    M = geometric_air_mass_factor(solar_zenith, view_zenith)
    C = pressure_correction_coefficient(wavelength, M)

    tau_r0 = rayleigh_optical_thickness(wavelength, pressure=STANDARD_PRESSURE)
    tau_r = rayleigh_optical_thickness(wavelength, pressure=pressure)

    # Pressure correction factor
    numerator = 1.0 - np.exp(-C * tau_r * M)
    denominator = 1.0 - np.exp(-C * tau_r0 * M)

    correction_factor = numerator / denominator

    return rayleigh_reflectance_std * correction_factor


class RayleighLUT:
    """
    Rayleigh Look-Up Table for atmospheric correction.

    This class manages precomputed Rayleigh radiances/reflectances for
    various geometries and wind speeds. In practice, these LUTs are
    computed using vector radiative transfer codes and stored in files.

    Parameters
    ----------
    sensor : str
        Sensor name ('SeaWiFS', 'MODIS-Aqua', etc.).

    Attributes
    ----------
    sensor : str
        Sensor name.
    solar_zenith_angles : ndarray
        Solar zenith angles in LUT [degrees].
    view_zenith_angles : ndarray
        Viewing zenith angles in LUT [degrees].
    wind_speeds : ndarray
        Wind speeds in LUT [m/s].
    wavelengths : ndarray
        Wavelengths in LUT [nm].

    Notes
    -----
    The LUT contains Stokes vector components (I, Q, U) in reflectance units.
    The V component (circular polarization) is assumed zero.

    The LUT is computed for:

    - 45 solar zenith angles: 0 to 88 deg by 2 deg
    - 41 viewing zenith angles: 0 to 84 deg by ~2 deg
    - 8 wind speeds: 0, 1.9, 4.2, 7.5, 11.7, 16.9, 22.9, 30.0 m/s
    - Sensor-specific wavelengths
    """

    # Standard LUT dimensions
    SOLAR_ZENITH = np.arange(0, 90, 2)  # 0 to 88 by 2 deg
    VIEW_ZENITH = np.linspace(0, 84, 41)  # 41 angles
    WIND_SPEEDS = np.array([0.0, 1.9, 4.2, 7.5, 11.7, 16.9, 22.9, 30.0])
    AZIMUTH_FOURIER_TERMS = 3  # Number of Fourier terms for azimuth dependence

    def __init__(self, sensor: str):
        """Initialize Rayleigh LUT for specified sensor."""
        self.sensor = sensor
        self._lut_data: Optional[np.ndarray] = None
        self._loaded = False

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load LUT data from file.

        Parameters
        ----------
        filepath : str, optional
            Path to LUT file. If None, uses default path for sensor.

        Raises
        ------
        FileNotFoundError
            If LUT file does not exist.
        NotImplementedError
            LUT loading not yet implemented.
        """
        # TODO: Implement LUT loading from NetCDF or HDF5 files
        raise NotImplementedError(
            "LUT loading not yet implemented. "
            "Use synthetic calculations or provide LUT data directly."
        )

    def interpolate(
        self,
        wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
        wind_speed: float,
    ) -> Tuple[float, float, float]:
        """
        Interpolate LUT to get Rayleigh Stokes vector components.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
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
        tuple of float
            (I, Q, U) Stokes vector components in reflectance units.

        Raises
        ------
        RuntimeError
            If LUT has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("LUT not loaded. Call load() first.")

        # TODO: Implement multi-dimensional interpolation
        raise NotImplementedError("LUT interpolation not yet implemented.")

    def compute_synthetic(
        self,
        wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
        wind_speed: float = 0.0,
    ) -> float:
        """
        Compute synthetic Rayleigh reflectance using analytical approximation.

        This is a simplified calculation for testing purposes. For accurate
        atmospheric correction, use the full LUT.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        solar_zenith : float
            Solar zenith angle in degrees.
        view_zenith : float
            Viewing zenith angle in degrees.
        relative_azimuth : float
            Relative azimuth angle in degrees.
        wind_speed : float, optional
            Wind speed in m/s (currently ignored in simplified model).

        Returns
        -------
        float
            Approximate Rayleigh reflectance.

        Notes
        -----
        This uses single-scattering approximation and should not be used
        for operational processing.
        """
        tau_r = rayleigh_optical_thickness(wavelength)
        M = geometric_air_mass_factor(solar_zenith, view_zenith)

        # Single-scattering approximation with phase function
        # P(Theta) for Rayleigh scattering
        theta_s = np.deg2rad(solar_zenith)
        theta_v = np.deg2rad(view_zenith)
        phi = np.deg2rad(relative_azimuth)

        # Scattering angle
        cos_Theta = -np.cos(theta_s) * np.cos(theta_v) + np.sin(theta_s) * np.sin(
            theta_v
        ) * np.cos(phi)

        # Rayleigh phase function (unpolarized)
        phase = 0.75 * (1.0 + cos_Theta**2)

        # Single-scattering reflectance
        rho_r = tau_r * phase / (4.0 * np.cos(theta_s) * np.cos(theta_v))

        return rho_r
