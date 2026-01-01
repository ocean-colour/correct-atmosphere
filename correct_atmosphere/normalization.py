"""
Normalized reflectances and BRDF corrections.

This module implements the calculation of normalized water-leaving
radiances and reflectances as described in Chapter 3 of Mobley et al. (2016),
including:

- Normalized water-leaving radiance [Lw]_N
- Exact normalized water-leaving reflectance [rho_w]_N^ex
- BRDF correction factors (f/Q, R)
- Remote-sensing reflectance Rrs

References
----------
.. [1] Gordon, H.R., et al. (1988). A semianalytical radiance model of ocean
       color. J. Geophys. Res., 93:10909-10924.
.. [2] Morel, A. and Gentili, B. (1996). Diffuse reflectance of oceanic waters:
       III: Implication of bidirectionality for the remote-sensing problem.
       Applied Optics, 35:4850-4862.
.. [3] Morel, A., et al. (2002). Bidirectional reflectance of oceanic waters:
       accounting for Raman emission and varying particle scattering phase
       function. Applied Optics, 41:6289-6306.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict

from oceanatmos.constants import (
    BRDF_R0,
    BRDF_WAVELENGTHS,
    BRDF_CHL_VALUES,
    BRDF_SOLAR_ZENITH,
    WATER_REFRACTIVE_INDEX,
)


def earth_sun_distance_correction(day_of_year: int) -> float:
    """
    Calculate Earth-Sun distance correction factor.

    Parameters
    ----------
    day_of_year : int
        Day of year (1-366).

    Returns
    -------
    float
        Correction factor (R/R0)^2.

    Notes
    -----
    The Earth-Sun distance varies by about ±1.7% over the year due to
    Earth's elliptical orbit. The correction factor (R/R0)^2 varies
    by about ±3.4%, or roughly 8% total range.

    The factor is used to normalize radiances to the mean Earth-Sun distance.
    """
    # Approximate formula for Earth-Sun distance
    # R/R0 = 1 - 0.01673 * cos(2*pi*(day - 4)/365.25)
    angle = 2.0 * np.pi * (day_of_year - 4) / 365.25
    r_ratio = 1.0 - 0.01673 * np.cos(angle)

    return r_ratio ** 2


def normalized_water_leaving_radiance(
    lw: Union[float, np.ndarray],
    solar_zenith: float,
    diffuse_transmittance: Union[float, np.ndarray],
    day_of_year: int = 172,  # Summer solstice default
) -> Union[float, np.ndarray]:
    """
    Calculate normalized water-leaving radiance [Lw]_N.

    Parameters
    ----------
    lw : float or array_like
        Water-leaving radiance at sea surface [W/(m^2 sr nm)].
    solar_zenith : float
        Solar zenith angle in degrees.
    diffuse_transmittance : float or array_like
        Diffuse atmospheric transmittance in sun direction.
    day_of_year : int, optional
        Day of year for Earth-Sun distance correction (default: 172).

    Returns
    -------
    float or ndarray
        Normalized water-leaving radiance [Lw]_N.

    Notes
    -----
    Implements Equation 3.2 from Mobley et al. (2016):

    .. math::

        [L_w]_N = \\left(\\frac{R}{R_0}\\right)^2 
                  \\frac{L_w(\\theta_s)}{\\cos\\theta_s \\cdot t(\\theta_s)}

    This normalization removes the effects of:

    - Earth-Sun distance variation
    - Solar zenith angle
    - Atmospheric attenuation

    Examples
    --------
    >>> lw_n = normalized_water_leaving_radiance(0.5, 30.0, 0.9)
    >>> print(f"Normalized Lw: {lw_n:.4f}")
    """
    cos_theta_s = np.cos(np.deg2rad(solar_zenith))
    r_correction = earth_sun_distance_correction(day_of_year)

    lw_n = r_correction * np.asarray(lw) / (cos_theta_s * diffuse_transmittance)

    return lw_n


def normalized_water_leaving_reflectance(
    lw: Union[float, np.ndarray],
    solar_irradiance: Union[float, np.ndarray],
    solar_zenith: float,
    diffuse_transmittance: Union[float, np.ndarray],
    day_of_year: int = 172,
) -> Union[float, np.ndarray]:
    """
    Calculate normalized water-leaving reflectance [rho_w]_N.

    Parameters
    ----------
    lw : float or array_like
        Water-leaving radiance at sea surface [W/(m^2 sr nm)].
    solar_irradiance : float or array_like
        Extraterrestrial solar irradiance F0 [W/(m^2 nm)].
    solar_zenith : float
        Solar zenith angle in degrees.
    diffuse_transmittance : float or array_like
        Diffuse atmospheric transmittance in sun direction.
    day_of_year : int, optional
        Day of year (default: 172).

    Returns
    -------
    float or ndarray
        Normalized water-leaving reflectance [rho_w]_N (dimensionless).

    Notes
    -----
    Implements Equation 3.3 from Mobley et al. (2016):

    .. math::

        [\\rho_w]_N = \\frac{\\pi}{F_0} [L_w]_N
                    = \\pi \\left(\\frac{R}{R_0}\\right)^2 
                      \\frac{L_w}{F_0 \\cos\\theta_s \\cdot t(\\theta_s)}

    The factor π has units of steradian, making [rho_w]_N dimensionless.
    """
    lw_n = normalized_water_leaving_radiance(
        lw, solar_zenith, diffuse_transmittance, day_of_year
    )

    rho_w_n = np.pi * lw_n / np.asarray(solar_irradiance)

    return rho_w_n


def remote_sensing_reflectance(
    lw: Union[float, np.ndarray],
    ed: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate remote-sensing reflectance Rrs.

    Parameters
    ----------
    lw : float or array_like
        Water-leaving radiance [W/(m^2 sr nm)].
    ed : float or array_like
        Downwelling irradiance at sea surface [W/(m^2 nm)].

    Returns
    -------
    float or ndarray
        Remote-sensing reflectance Rrs [sr^-1].

    Notes
    -----
    Implements Equation 3.4 from Mobley et al. (2016):

    .. math::

        R_{rs} = \\frac{L_w}{E_d(0^+)}

    Note that [rho_w]_N = pi * Rrs when both refer to the same conditions.

    Examples
    --------
    >>> rrs = remote_sensing_reflectance(0.5, 500.0)
    >>> print(f"Rrs: {rrs:.6f} sr^-1")
    """
    return np.asarray(lw) / np.asarray(ed)


def rrs_to_normalized_reflectance(
    rrs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert remote-sensing reflectance to normalized reflectance.

    Parameters
    ----------
    rrs : float or array_like
        Remote-sensing reflectance [sr^-1].

    Returns
    -------
    float or ndarray
        Normalized water-leaving reflectance [rho_w]_N (dimensionless).

    Notes
    -----
    [rho_w]_N = pi * Rrs

    See Equation 3.6 in Mobley et al. (2016).
    """
    return np.pi * np.asarray(rrs)


def normalized_reflectance_to_rrs(
    rho_w_n: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert normalized reflectance to remote-sensing reflectance.

    Parameters
    ----------
    rho_w_n : float or array_like
        Normalized water-leaving reflectance [rho_w]_N (dimensionless).

    Returns
    -------
    float or ndarray
        Remote-sensing reflectance Rrs [sr^-1].
    """
    return np.asarray(rho_w_n) / np.pi


def surface_transmission_factor_R(
    view_zenith_water: float,
    wind_speed: float = 0.0,
) -> float:
    """
    Calculate surface transmission factor R(theta'_v, W).

    This factor accounts for all transmission and reflection effects by
    the wind-blown sea surface for both downward (Ed) and upward (Lu)
    paths through the surface.

    Parameters
    ----------
    view_zenith_water : float
        In-water viewing zenith angle theta'_v in degrees.
    wind_speed : float, optional
        Wind speed in m/s (default: 0).

    Returns
    -------
    float
        Surface transmission factor R (dimensionless).

    Notes
    -----
    R ≈ 0.529 for normal incidence (theta'_v = 0).
    The factor has weak dependence on wind speed (Gordon, 2005).

    The reference value R0 = R(0, W) is used in the BRDF correction.
    """
    # Simplified model: R decreases with viewing angle
    # Based on Morel et al. (2002) Figure 4
    theta = np.deg2rad(view_zenith_water)

    # Approximate formula
    R = BRDF_R0 * np.cos(theta) ** 0.2

    return R


def fresnel_transmittance_water_to_air(
    theta_water: float,
    n_water: float = WATER_REFRACTIVE_INDEX,
) -> float:
    """
    Calculate Fresnel transmittance from water to air.

    Parameters
    ----------
    theta_water : float
        Angle in water in degrees (from normal).
    n_water : float, optional
        Water refractive index (default: 1.34).

    Returns
    -------
    float
        Fresnel transmittance (0 to 1).
    """
    theta_w = np.deg2rad(theta_water)

    # Check for total internal reflection
    sin_theta_air = n_water * np.sin(theta_w)
    if sin_theta_air >= 1.0:
        return 0.0

    theta_air = np.arcsin(sin_theta_air)

    # Fresnel equations
    cos_w = np.cos(theta_w)
    cos_a = np.cos(theta_air)

    rs = ((n_water * cos_w - cos_a) / (n_water * cos_w + cos_a)) ** 2
    rp = ((cos_w - n_water * cos_a) / (cos_w + n_water * cos_a)) ** 2

    reflectance = 0.5 * (rs + rp)
    transmittance = 1.0 - reflectance

    # Account for refraction solid angle change
    transmittance *= (cos_a / cos_w) / (n_water ** 2)

    return transmittance


def snell_angle(
    theta_air: float,
    n_water: float = WATER_REFRACTIVE_INDEX,
) -> float:
    """
    Calculate in-water angle from in-air angle using Snell's law.

    Parameters
    ----------
    theta_air : float
        In-air angle in degrees.
    n_water : float, optional
        Water refractive index (default: 1.34).

    Returns
    -------
    float
        In-water angle theta'_v in degrees.
    """
    sin_theta_air = np.sin(np.deg2rad(theta_air))
    sin_theta_water = sin_theta_air / n_water

    if sin_theta_water > 1:
        # Total internal reflection limit
        return 90.0

    return np.rad2deg(np.arcsin(sin_theta_water))


class BRDFCorrection:
    """
    BRDF correction factors for normalized water-leaving radiance.

    This class provides methods to compute the f/Q and R factors used
    to correct water-leaving radiance for bidirectional effects.

    Parameters
    ----------
    include_raman : bool, optional
        Include Raman scattering effects (default: True).

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths in f/Q tables [nm].
    chl_values : ndarray
        Chlorophyll values in f/Q tables [mg/m^3].
    solar_zenith_values : ndarray
        Solar zenith angles in f/Q tables [degrees].

    Notes
    -----
    The BRDF correction converts a measurement made for a particular
    viewing geometry to what would be measured for a zenith sun and
    nadir viewing direction.

    The correction is:

    .. math::

        [L_w]_N^{ex} = [L_w]_N \\frac{R_0}{R} \\frac{f_0/Q_0}{f/Q}

    The f/Q tables are computed for Case 1 waters where IOPs can be
    parameterized by chlorophyll concentration.
    """

    def __init__(self, include_raman: bool = True):
        """Initialize BRDF correction with Morel et al. (2002) tables."""
        self.include_raman = include_raman
        self.wavelengths = np.array(BRDF_WAVELENGTHS)
        self.chl_values = np.array(BRDF_CHL_VALUES)
        self.solar_zenith_values = np.array(BRDF_SOLAR_ZENITH)
        self._loaded = False

    def load_tables(self, filepath: Optional[str] = None) -> None:
        """
        Load f/Q tables from file.

        Parameters
        ----------
        filepath : str, optional
            Path to f/Q table file.
        """
        # TODO: Load from ftp://oceane.obs-vlfr.fr/pub/gentili/AppliedOptics2002/
        raise NotImplementedError("f/Q table loading not yet implemented.")

    def get_f_over_Q(
        self,
        wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
        chlorophyll: float,
        wind_speed: float = 0.0,
    ) -> float:
        """
        Get f/Q factor for given conditions.

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
        chlorophyll : float
            Chlorophyll concentration in mg/m^3.
        wind_speed : float, optional
            Wind speed in m/s (default: 0).

        Returns
        -------
        float
            f/Q factor (typically 0.07 to 0.15).
        """
        if self._loaded:
            # TODO: Interpolate from loaded tables
            raise NotImplementedError("Table interpolation not yet implemented.")

        # Synthetic approximation based on Morel et al. (2002)
        return self._synthetic_f_Q(
            wavelength, solar_zenith, view_zenith, relative_azimuth, chlorophyll
        )

    def _synthetic_f_Q(
        self,
        wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
        chlorophyll: float,
    ) -> float:
        """Generate synthetic f/Q value for testing."""
        # Simplified model based on typical behavior
        # f/Q ≈ 0.09 for nadir viewing, decreasing off-nadir

        # Base value
        f_Q_nadir = 0.09

        # Viewing angle effect
        theta_v = np.deg2rad(view_zenith)
        view_factor = 1.0 - 0.3 * np.sin(theta_v) ** 2

        # Solar zenith effect
        theta_s = np.deg2rad(solar_zenith)
        solar_factor = 1.0 - 0.1 * (1.0 - np.cos(theta_s))

        # Chlorophyll effect (higher Chl = more anisotropic)
        chl_factor = 1.0 + 0.05 * np.log10(chlorophyll + 0.03)

        f_Q = f_Q_nadir * view_factor * solar_factor * chl_factor

        return max(0.05, min(0.15, f_Q))

    def get_f0_over_Q0(
        self,
        wavelength: float,
        chlorophyll: float,
        wind_speed: float = 0.0,
    ) -> float:
        """
        Get reference f0/Q0 for zenith sun and nadir viewing.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        chlorophyll : float
            Chlorophyll concentration in mg/m^3.
        wind_speed : float, optional
            Wind speed in m/s.

        Returns
        -------
        float
            Reference f0/Q0 factor.
        """
        return self.get_f_over_Q(wavelength, 0.0, 0.0, 0.0, chlorophyll, wind_speed)

    def correction_factor(
        self,
        wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
        chlorophyll: float,
        wind_speed: float = 0.0,
    ) -> float:
        """
        Calculate complete BRDF correction factor.

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
        chlorophyll : float
            Chlorophyll concentration in mg/m^3.
        wind_speed : float, optional
            Wind speed in m/s (default: 0).

        Returns
        -------
        float
            BRDF correction factor (R0/R) * (f0/Q0) / (f/Q).

        Notes
        -----
        Implements Equation 3.7 from Mobley et al. (2016).
        Values are typically in the range 0.6 to 1.2.
        """
        # Get in-water angle
        theta_v_water = snell_angle(view_zenith)

        # R factors
        R = surface_transmission_factor_R(theta_v_water, wind_speed)
        R0 = surface_transmission_factor_R(0.0, wind_speed)

        # f/Q factors
        f_Q = self.get_f_over_Q(
            wavelength, solar_zenith, view_zenith, relative_azimuth, chlorophyll, wind_speed
        )
        f0_Q0 = self.get_f0_over_Q0(wavelength, chlorophyll, wind_speed)

        # Combined correction
        correction = (R0 / R) * (f0_Q0 / f_Q)

        return correction


def exact_normalized_reflectance(
    rho_w: Union[float, np.ndarray],
    wavelength: Union[float, np.ndarray],
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
    chlorophyll: float,
    solar_irradiance: Union[float, np.ndarray],
    diffuse_transmittance: Union[float, np.ndarray],
    wind_speed: float = 0.0,
    day_of_year: int = 172,
    brdf_correction: Optional[BRDFCorrection] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate exact normalized water-leaving reflectance [rho_w]_N^ex.

    This is the standard product reported by NASA OBPG as "remote-sensing
    reflectance" (after dividing by pi).

    Parameters
    ----------
    rho_w : float or array_like
        Water-leaving reflectance at sea surface.
    wavelength : float or array_like
        Wavelength in nm.
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle in degrees.
    chlorophyll : float
        Chlorophyll concentration in mg/m^3 (for BRDF correction).
    solar_irradiance : float or array_like
        Extraterrestrial solar irradiance F0 [W/(m^2 nm)].
    diffuse_transmittance : float or array_like
        Diffuse atmospheric transmittance in sun direction.
    wind_speed : float, optional
        Wind speed in m/s (default: 0).
    day_of_year : int, optional
        Day of year (default: 172).
    brdf_correction : BRDFCorrection, optional
        BRDF correction object.

    Returns
    -------
    float or ndarray
        Exact normalized water-leaving reflectance [rho_w]_N^ex.

    Notes
    -----
    Implements Equations 3.7-3.9 from Mobley et al. (2016):

    .. math::

        [\\rho_w]_N^{ex} = \\frac{\\pi}{F_0 \\cos\\theta_s t(\\theta_s)}
                          \\frac{R_0}{R} \\frac{f_0/Q_0}{f/Q} L_w

    The NASA OBPG "Rrs" is [rho_w]_N^ex / pi.
    """
    if brdf_correction is None:
        brdf_correction = BRDFCorrection()

    wavelength = np.asarray(wavelength)
    rho_w = np.asarray(rho_w)

    # Earth-Sun distance correction
    r_correction = earth_sun_distance_correction(day_of_year)

    # Basic normalization
    cos_theta_s = np.cos(np.deg2rad(solar_zenith))
    rho_w_n = r_correction * rho_w / (cos_theta_s * diffuse_transmittance)

    # BRDF correction for each wavelength
    if wavelength.ndim == 0:
        wavelengths = [float(wavelength)]
    else:
        wavelengths = wavelength.tolist()

    brdf_factors = np.array([
        brdf_correction.correction_factor(
            wl, solar_zenith, view_zenith, relative_azimuth, chlorophyll, wind_speed
        )
        for wl in wavelengths
    ])

    if wavelength.ndim == 0:
        brdf_factors = brdf_factors[0]

    rho_w_n_ex = rho_w_n * brdf_factors

    return rho_w_n_ex


def nasa_rrs(
    rho_w_n_ex: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert exact normalized reflectance to NASA Rrs product.

    Parameters
    ----------
    rho_w_n_ex : float or array_like
        Exact normalized water-leaving reflectance.

    Returns
    -------
    float or ndarray
        NASA OBPG "Rrs" [sr^-1].

    Notes
    -----
    Implements Equation 3.10 from Mobley et al. (2016):

    .. math::

        R_{rs}(\\text{NASA}) = \\frac{[\\rho_w]_N^{ex}}{\\pi}
    """
    return np.asarray(rho_w_n_ex) / np.pi
