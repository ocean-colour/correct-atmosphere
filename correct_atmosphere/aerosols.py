"""
Aerosol path radiance estimation algorithms.

This module implements the aerosol correction algorithms described in
Chapter 9 of Mobley et al. (2016), including:

- Aerosol optical properties (particle size distributions, optical thickness)
- Black-pixel algorithm for clear oceanic waters
- Non-black-pixel algorithm for turbid/productive waters
- Aerosol model selection based on epsilon ratios

References
----------
.. [1] Gordon, H.R. and Wang, M. (1994). Retrieval of water-leaving radiance
       and aerosol optical thickness over the oceans with SeaWiFS: a
       preliminary algorithm. Applied Optics, 33:443-452.
.. [2] Ahmad, Z., et al. (2010). New aerosol models for the retrieval of
       aerosol optical thickness and normalized water-leaving radiances
       from SeaWiFS and MODIS sensors. Applied Optics, 49:5545-5560.
.. [3] Bailey, S.W., et al. (2010). Estimation of near-infrared water-leaving
       reflectance for satellite ocean color data processing. Optics Express,
       18:7521-7527.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict, List
from dataclasses import dataclass

from correct_atmosphere.constants import (
    AEROSOL_RH_VALUES,
    AEROSOL_NUM_MODELS,
    NIR_BANDS,
    NONBLACK_PIXEL_CHL_MIN,
    NONBLACK_PIXEL_CHL_MAX,
    NONBLACK_PIXEL_CONVERGENCE,
    NONBLACK_PIXEL_MAX_ITER,
    PURE_WATER_ABSORPTION,
    PURE_WATER_BACKSCATTER,
)


@dataclass
class AerosolModel:
    """
    Aerosol model parameters.

    Attributes
    ----------
    model_id : int
        Model identifier (0-9).
    fine_fraction : float
        Fraction of fine mode particles (0-1).
    angstrom_exponent : float
        Angstrom exponent alpha.
    effective_radius : float
        Effective radius r_eff in micrometers.
    relative_humidity : float
        Relative humidity in percent.
    """

    model_id: int
    fine_fraction: float
    angstrom_exponent: float
    effective_radius: float
    relative_humidity: float


def angstrom_exponent(
    tau_1: float,
    tau_2: float,
    wavelength_1: float,
    wavelength_2: float,
) -> float:
    """
    Calculate Angstrom exponent from aerosol optical thickness at two wavelengths.

    Parameters
    ----------
    tau_1 : float
        Aerosol optical thickness at wavelength_1.
    tau_2 : float
        Aerosol optical thickness at wavelength_2.
    wavelength_1 : float
        First wavelength in nm.
    wavelength_2 : float
        Second wavelength in nm (reference wavelength).

    Returns
    -------
    float
        Angstrom exponent alpha.

    Notes
    -----
    Implements Equation 9.1 from Mobley et al. (2016):

    .. math::

        \\frac{\\tau_a(\\lambda)}{\\tau_a(\\lambda_0)} = 
        \\left(\\frac{\\lambda_0}{\\lambda}\\right)^\\alpha

    Rearranging:

    .. math::

        \\alpha = -\\frac{\\ln(\\tau_1/\\tau_2)}{\\ln(\\lambda_1/\\lambda_2)}

    Examples
    --------
    >>> alpha = angstrom_exponent(0.15, 0.10, 443.0, 865.0)
    >>> print(f"Angstrom exponent: {alpha:.2f}")
    """
    if tau_1 <= 0 or tau_2 <= 0:
        return 0.0

    alpha = -np.log(tau_1 / tau_2) / np.log(wavelength_1 / wavelength_2)

    return alpha


def aerosol_optical_thickness(
    wavelength: Union[float, np.ndarray],
    reference_tau: float,
    reference_wavelength: float,
    alpha: float,
) -> Union[float, np.ndarray]:
    """
    Calculate aerosol optical thickness at given wavelength using Angstrom law.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength(s) in nm.
    reference_tau : float
        Aerosol optical thickness at reference wavelength.
    reference_wavelength : float
        Reference wavelength in nm.
    alpha : float
        Angstrom exponent.

    Returns
    -------
    float or ndarray
        Aerosol optical thickness at requested wavelength(s).

    Notes
    -----
    Implements Equation 9.1:

    .. math::

        \\tau_a(\\lambda) = \\tau_a(\\lambda_0) 
        \\left(\\frac{\\lambda_0}{\\lambda}\\right)^\\alpha
    """
    wavelength = np.asarray(wavelength)

    tau = reference_tau * (reference_wavelength / wavelength) ** alpha

    return tau


def epsilon_ratio(
    rho_a_lambda1: float,
    rho_a_lambda2: float,
) -> float:
    """
    Calculate aerosol reflectance ratio epsilon.

    Parameters
    ----------
    rho_a_lambda1 : float
        Aerosol reflectance at shorter wavelength lambda_1.
    rho_a_lambda2 : float
        Aerosol reflectance at longer wavelength lambda_2.

    Returns
    -------
    float
        Epsilon ratio = rho_a(lambda_1) / rho_a(lambda_2).

    Notes
    -----
    Implements Equation 9.4 from Mobley et al. (2016):

    .. math::

        \\epsilon(\\lambda_1, \\lambda_2) = 
        \\frac{\\rho_A(\\lambda_1)}{\\rho_A(\\lambda_2)}

    The epsilon ratio is used to select the appropriate aerosol model.
    """
    if rho_a_lambda2 <= 0:
        return 0.0

    return rho_a_lambda1 / rho_a_lambda2


class AerosolLUT:
    """
    Aerosol Look-Up Table for atmospheric correction.

    This class manages precomputed aerosol reflectances and epsilon
    ratios for various aerosol models and geometries.

    Parameters
    ----------
    sensor : str
        Sensor name ('SeaWiFS', 'MODIS-Aqua', etc.).

    Attributes
    ----------
    sensor : str
        Sensor name.
    num_models : int
        Number of aerosol models (default: 10).
    rh_values : tuple
        Relative humidity values in LUT.

    Notes
    -----
    The LUT contains:

    - Single-scattering aerosol reflectance ratios for each band
    - Quadratic coefficients for single-to-multiple scattering conversion
    - Diffuse transmittance coefficients t = A exp(-B tau_a)

    The 80 aerosol tables (10 models × 8 RH values) are from
    Ahmad et al. (2010).
    """

    def __init__(self, sensor: str):
        """Initialize aerosol LUT for specified sensor."""
        self.sensor = sensor
        self.num_models = AEROSOL_NUM_MODELS
        self.rh_values = AEROSOL_RH_VALUES
        self._loaded = False
        self._epsilon_tables: Optional[Dict] = None

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load LUT data from file.

        Parameters
        ----------
        filepath : str, optional
            Path to LUT file. If None, uses default path for sensor.

        Raises
        ------
        NotImplementedError
            LUT loading not yet implemented.
        """
        # TODO: Implement LUT loading from NetCDF/HDF5
        raise NotImplementedError(
            "LUT loading not yet implemented. "
            "Use synthetic calculations or provide LUT data directly."
        )

    def get_epsilon(
        self,
        model_id: int,
        relative_humidity: float,
        wavelength: float,
        reference_wavelength: float,
        solar_zenith: float,
        view_zenith: float,
        relative_azimuth: float,
    ) -> float:
        """
        Get epsilon ratio from LUT for specified conditions.

        Parameters
        ----------
        model_id : int
            Aerosol model ID (0-9).
        relative_humidity : float
            Relative humidity in percent.
        wavelength : float
            Wavelength in nm.
        reference_wavelength : float
            Reference wavelength in nm (lambda_2).
        solar_zenith : float
            Solar zenith angle in degrees.
        view_zenith : float
            Viewing zenith angle in degrees.
        relative_azimuth : float
            Relative azimuth angle in degrees.

        Returns
        -------
        float
            Epsilon ratio epsilon(wavelength, reference_wavelength).
        """
        if not self._loaded:
            # Return synthetic estimate
            return self._synthetic_epsilon(
                model_id,
                wavelength,
                reference_wavelength,
            )

        # TODO: Implement LUT interpolation
        raise NotImplementedError("LUT interpolation not yet implemented.")

    def _synthetic_epsilon(
        self,
        model_id: int,
        wavelength: float,
        reference_wavelength: float,
    ) -> float:
        """
        Generate synthetic epsilon for testing.

        Based on typical aerosol model behavior shown in Figure 9.3.
        """
        # Model parameters (approximate values from Ahmad et al. 2010)
        # Higher model_id = more fine particles = larger Angstrom exponent
        alpha_values = [
            -0.123, 0.020, 0.144, 0.443, 0.788,
            1.202, 1.447, 1.732, 1.950, 2.017
        ]
        alpha = alpha_values[model_id]

        # Epsilon approximately follows (lambda_2/lambda)^alpha
        epsilon = (reference_wavelength / wavelength) ** alpha

        return epsilon


def black_pixel_correction(
    rho_aw_nir1: float,
    rho_aw_nir2: float,
    wavelengths: np.ndarray,
    nir_wavelength_1: float,
    nir_wavelength_2: float,
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
    relative_humidity: float,
    aerosol_lut: Optional[AerosolLUT] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Perform black-pixel aerosol correction.

    This algorithm assumes water-leaving radiance is zero at NIR wavelengths,
    valid for low-chlorophyll Case 1 waters.

    Parameters
    ----------
    rho_aw_nir1 : float
        TOA reflectance (minus Rayleigh, glint, whitecaps) at NIR band 1.
    rho_aw_nir2 : float
        TOA reflectance (minus Rayleigh, glint, whitecaps) at NIR band 2.
    wavelengths : ndarray
        Wavelengths at which to compute aerosol reflectance.
    nir_wavelength_1 : float
        NIR band 1 wavelength (lambda_1, shorter).
    nir_wavelength_2 : float
        NIR band 2 wavelength (lambda_2, longer).
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle in degrees.
    relative_humidity : float
        Relative humidity in percent.
    aerosol_lut : AerosolLUT, optional
        Aerosol lookup table. If None, uses synthetic calculations.

    Returns
    -------
    tuple
        (rho_a, info) where:
        - rho_a: ndarray of aerosol reflectance at each wavelength
        - info: dict with 'model_low', 'model_high', 'delta', 'epsilon_measured'

    Notes
    -----
    Implements the algorithm from Section 9.2 of Mobley et al. (2016):

    1. Under black-pixel assumption, rho_Aw(NIR) = rho_A(NIR)
    2. Compute measured epsilon = rho_A(lambda_1) / rho_A(lambda_2)
    3. Find aerosol models bracketing measured epsilon
    4. Interpolate to get rho_A at all wavelengths

    Examples
    --------
    >>> wavelengths = np.array([412, 443, 490, 555, 670])
    >>> rho_a, info = black_pixel_correction(
    ...     0.015, 0.012, wavelengths, 765.0, 865.0,
    ...     30.0, 15.0, 90.0, 80.0
    ... )
    """
    if aerosol_lut is None:
        aerosol_lut = AerosolLUT("generic")

    # Under black-pixel assumption: rho_Aw = rho_A at NIR
    rho_a_nir1 = rho_aw_nir1
    rho_a_nir2 = rho_aw_nir2

    # Measured epsilon ratio
    eps_measured = epsilon_ratio(rho_a_nir1, rho_a_nir2)

    # Find bracketing aerosol models
    # Search through models to find those with epsilon_low < eps_measured < epsilon_high
    model_low = 0
    model_high = AEROSOL_NUM_MODELS - 1
    eps_low = 0.0
    eps_high = 10.0

    for model_id in range(AEROSOL_NUM_MODELS):
        eps_model = aerosol_lut.get_epsilon(
            model_id,
            relative_humidity,
            nir_wavelength_1,
            nir_wavelength_2,
            solar_zenith,
            view_zenith,
            relative_azimuth,
        )

        if eps_model <= eps_measured:
            if eps_model > eps_low:
                eps_low = eps_model
                model_low = model_id
        else:
            if eps_model < eps_high:
                eps_high = eps_model
                model_high = model_id

    # Interpolation factor (Equation 9.5)
    if eps_high > eps_low:
        delta = (eps_measured - eps_low) / (eps_high - eps_low)
    else:
        delta = 0.5

    delta = np.clip(delta, 0.0, 1.0)

    # Compute aerosol reflectance at all wavelengths (Equation 9.6)
    rho_a = np.zeros_like(wavelengths, dtype=float)

    for i, wl in enumerate(wavelengths):
        eps_low_wl = aerosol_lut.get_epsilon(
            model_low,
            relative_humidity,
            wl,
            nir_wavelength_2,
            solar_zenith,
            view_zenith,
            relative_azimuth,
        )
        eps_high_wl = aerosol_lut.get_epsilon(
            model_high,
            relative_humidity,
            wl,
            nir_wavelength_2,
            solar_zenith,
            view_zenith,
            relative_azimuth,
        )

        # Interpolate epsilon
        eps_wl = (1 - delta) * eps_low_wl + delta * eps_high_wl

        # Aerosol reflectance at this wavelength
        rho_a[i] = eps_wl * rho_a_nir2

    info = {
        "model_low": model_low,
        "model_high": model_high,
        "delta": delta,
        "epsilon_measured": eps_measured,
        "epsilon_low": eps_low,
        "epsilon_high": eps_high,
    }

    return rho_a, info


def estimate_nir_rrs(
    rrs_443: float,
    rrs_555: float,
    rrs_670: float,
    nir_wavelength: float,
) -> float:
    """
    Estimate remote-sensing reflectance at NIR wavelength.

    Used in non-black-pixel algorithm to estimate water-leaving
    contribution at NIR bands.

    Parameters
    ----------
    rrs_443 : float
        Remote-sensing reflectance at 443 nm [sr^-1].
    rrs_555 : float
        Remote-sensing reflectance at 555 nm [sr^-1].
    rrs_670 : float
        Remote-sensing reflectance at 670 nm [sr^-1].
    nir_wavelength : float
        NIR wavelength in nm.

    Returns
    -------
    float
        Estimated Rrs at NIR wavelength [sr^-1].

    Notes
    -----
    Implements the algorithm from Bailey et al. (2010):

    1. Calculate eta from Rrs(443)/Rrs(555) (Eq. 9.10)
    2. Get absorption a(670) from chlorophyll (Eq. 9.11)
    3. Solve for bb(670) using Rrs(670) (Eq. 9.9)
    4. Extrapolate bb to NIR using eta (Eq. 9.12)
    5. Calculate Rrs(NIR) from bb(NIR) and a_w(NIR)
    """
    # Step 1: Calculate eta (Eq. 9.10)
    if rrs_555 > 0 and rrs_443 > 0:
        ratio = rrs_443 / rrs_555
        eta = 2.0 * (1.0 - 1.2 * np.exp(-0.9 * ratio))
    else:
        eta = 1.0  # Default value

    # Step 2: Estimate chlorophyll from band ratio (simplified OC4 type)
    if rrs_555 > 0:
        # Very simplified chlorophyll estimate
        log_ratio = np.log10(max(rrs_443, 1e-6) / max(rrs_555, 1e-6))
        chl = 10 ** (0.366 - 3.067 * log_ratio)
        chl = np.clip(chl, 0.01, 100.0)
    else:
        chl = 0.1

    # Step 3: Get a(670) from chlorophyll (Eq. 9.11)
    a_w_670 = PURE_WATER_ABSORPTION.get(670.0, 0.439)
    a_670 = np.exp(0.9389 * np.log(chl) - 3.7589) + a_w_670

    # Step 4: Solve for bb(670) using Gordon model (Eq. 9.9)
    # Rrs ≈ (f/Q) * bb / (a + bb)
    # Approximate f/Q ≈ 0.09 for typical conditions
    f_Q = 0.09

    if rrs_670 > 0:
        # Solve: Rrs = f_Q * bb / (a + bb)
        # bb = Rrs * a / (f_Q - Rrs)
        denom = f_Q - rrs_670
        if denom > 0:
            bb_670 = rrs_670 * a_670 / denom
        else:
            bb_670 = 0.001
    else:
        bb_670 = 0.001

    bb_w_670 = PURE_WATER_BACKSCATTER.get(670.0, 0.000168)
    bb_p_670 = max(bb_670 - bb_w_670, 0.0)

    # Step 5: Extrapolate bb to NIR (Eq. 9.12)
    bb_w_nir = np.interp(
        nir_wavelength,
        list(PURE_WATER_BACKSCATTER.keys()),
        list(PURE_WATER_BACKSCATTER.values()),
    )
    bb_p_nir = bb_p_670 * (670.0 / nir_wavelength) ** eta
    bb_nir = bb_w_nir + bb_p_nir

    # Step 6: Calculate Rrs(NIR)
    a_w_nir = np.interp(
        nir_wavelength,
        list(PURE_WATER_ABSORPTION.keys()),
        list(PURE_WATER_ABSORPTION.values()),
    )
    # At NIR, absorption dominated by water
    a_nir = a_w_nir

    rrs_nir = f_Q * bb_nir / (a_nir + bb_nir)

    return max(rrs_nir, 0.0)


def non_black_pixel_correction(
    rho_aw_bands: np.ndarray,
    wavelengths: np.ndarray,
    nir_wavelength_1: float,
    nir_wavelength_2: float,
    solar_zenith: float,
    view_zenith: float,
    relative_azimuth: float,
    relative_humidity: float,
    diffuse_transmittance: np.ndarray,
    aerosol_lut: Optional[AerosolLUT] = None,
    max_iterations: int = NONBLACK_PIXEL_MAX_ITER,
    convergence: float = NONBLACK_PIXEL_CONVERGENCE,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform non-black-pixel aerosol correction.

    This iterative algorithm estimates water-leaving radiance at NIR
    wavelengths for turbid or productive waters where the black-pixel
    assumption is not valid.

    Parameters
    ----------
    rho_aw_bands : ndarray
        TOA reflectance (minus Rayleigh, glint, whitecaps) at all bands.
    wavelengths : ndarray
        Wavelengths in nm corresponding to rho_aw_bands.
    nir_wavelength_1 : float
        NIR band 1 wavelength (shorter).
    nir_wavelength_2 : float
        NIR band 2 wavelength (longer).
    solar_zenith : float
        Solar zenith angle in degrees.
    view_zenith : float
        Viewing zenith angle in degrees.
    relative_azimuth : float
        Relative azimuth angle in degrees.
    relative_humidity : float
        Relative humidity in percent.
    diffuse_transmittance : ndarray
        Diffuse transmittance at each wavelength.
    aerosol_lut : AerosolLUT, optional
        Aerosol lookup table.
    max_iterations : int, optional
        Maximum iterations (default: 10).
    convergence : float, optional
        Convergence threshold for Rrs(NIR) change (default: 0.02 = 2%).

    Returns
    -------
    tuple
        (rho_a, rrs, info) where:
        - rho_a: ndarray of aerosol reflectance at each wavelength
        - rrs: ndarray of remote-sensing reflectance at each wavelength
        - info: dict with iteration details

    Notes
    -----
    Implements the algorithm from Section 9.3 of Mobley et al. (2016):

    1. Start with black-pixel assumption (Rrs_NIR = 0)
    2. Perform atmospheric correction to get Rrs
    3. Estimate Rrs(NIR) from visible bands
    4. Subtract water contribution from NIR TOA signal
    5. Repeat until Rrs(NIR) converges
    """
    if aerosol_lut is None:
        aerosol_lut = AerosolLUT("generic")

    # Find band indices
    idx_443 = np.argmin(np.abs(wavelengths - 443))
    idx_555 = np.argmin(np.abs(wavelengths - 555))
    idx_670 = np.argmin(np.abs(wavelengths - 670))
    idx_nir1 = np.argmin(np.abs(wavelengths - nir_wavelength_1))
    idx_nir2 = np.argmin(np.abs(wavelengths - nir_wavelength_2))

    # Initialize
    rrs_nir1 = 0.0
    rrs_nir2 = 0.0
    rho_w_nir1 = 0.0
    rho_w_nir2 = 0.0

    iteration_history = []

    for iteration in range(max_iterations):
        # Correct NIR for water contribution
        rho_aw_nir1_corrected = rho_aw_bands[idx_nir1] - rho_w_nir1
        rho_aw_nir2_corrected = rho_aw_bands[idx_nir2] - rho_w_nir2

        # Perform black-pixel correction with corrected NIR
        rho_a, bp_info = black_pixel_correction(
            rho_aw_nir1_corrected,
            rho_aw_nir2_corrected,
            wavelengths,
            nir_wavelength_1,
            nir_wavelength_2,
            solar_zenith,
            view_zenith,
            relative_azimuth,
            relative_humidity,
            aerosol_lut,
        )

        # Calculate Rrs from (rho_aw - rho_a) / t
        rho_w = (rho_aw_bands - rho_a) / diffuse_transmittance
        rrs = rho_w / np.pi  # Convert reflectance to Rrs

        # Ensure non-negative
        rrs = np.maximum(rrs, 0.0)

        # Estimate new NIR Rrs
        rrs_nir1_new = estimate_nir_rrs(
            rrs[idx_443], rrs[idx_555], rrs[idx_670], nir_wavelength_1
        )
        rrs_nir2_new = estimate_nir_rrs(
            rrs[idx_443], rrs[idx_555], rrs[idx_670], nir_wavelength_2
        )

        # Check convergence
        if rrs_nir1 > 0:
            change_nir1 = abs(rrs_nir1_new - rrs_nir1) / rrs_nir1
        else:
            change_nir1 = abs(rrs_nir1_new - rrs_nir1)

        iteration_history.append({
            "iteration": iteration,
            "rrs_nir1": rrs_nir1_new,
            "rrs_nir2": rrs_nir2_new,
            "change": change_nir1,
        })

        if iteration > 0 and change_nir1 < convergence:
            break

        # Update for next iteration
        rrs_nir1 = rrs_nir1_new
        rrs_nir2 = rrs_nir2_new
        rho_w_nir1 = rrs_nir1 * np.pi * diffuse_transmittance[idx_nir1]
        rho_w_nir2 = rrs_nir2 * np.pi * diffuse_transmittance[idx_nir2]

    info = {
        "iterations": len(iteration_history),
        "converged": len(iteration_history) < max_iterations,
        "history": iteration_history,
        "aerosol_info": bp_info,
    }

    return rho_a, rrs, info


def should_apply_nonblack_pixel(
    chlorophyll_estimate: float,
    chl_min: float = NONBLACK_PIXEL_CHL_MIN,
    chl_max: float = NONBLACK_PIXEL_CHL_MAX,
) -> Tuple[bool, float]:
    """
    Determine if non-black-pixel algorithm should be applied.

    Parameters
    ----------
    chlorophyll_estimate : float
        Initial chlorophyll estimate in mg/m^3.
    chl_min : float, optional
        Minimum Chl for possible correction (default: 0.3).
    chl_max : float, optional
        Chl above which correction is always applied (default: 0.7).

    Returns
    -------
    tuple
        (apply, weight) where:
        - apply: bool, whether to apply non-black-pixel
        - weight: float, interpolation weight (0 to 1) for transition zone

    Notes
    -----
    - Chl < 0.3: Never apply (weight = 0)
    - 0.3 <= Chl <= 0.7: Linear transition (weight = 0 to 1)
    - Chl > 0.7: Always apply (weight = 1)

    This prevents discontinuities in the retrieved products.
    """
    if chlorophyll_estimate < chl_min:
        return False, 0.0
    elif chlorophyll_estimate > chl_max:
        return True, 1.0
    else:
        weight = (chlorophyll_estimate - chl_min) / (chl_max - chl_min)
        return True, weight
