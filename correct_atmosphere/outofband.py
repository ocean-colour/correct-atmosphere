"""
Out-of-Band Spectral Response Correction.

This module implements the spectral out-of-band (OOB) correction algorithms
described in Section 10 of NASA TM-2016-217551 (Mobley et al., 2016).

Ocean color sensors have non-ideal spectral response functions with significant
sensitivity outside their nominal bandwidths. This "out-of-band" response can
cause ~1% errors in TOA radiance measurements, which translates to ~10% errors
in retrieved water-leaving radiance. Separate corrections are applied to
Rayleigh, aerosol, and water-leaving radiance components.

The correction converts full-band measurements to equivalent values for an
idealized sensor with a perfect "top hat" response over the nominal FWHM.

References
----------
.. [Gordon1995] Gordon, H.R. (1995). Remote sensing of ocean color: a methodology
   for dealing with broad spectral bands and significant out-of-band response.
   Applied Optics, 34, 8363-8374.

.. [Mobley2016] Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., & Bailey, S.
   (2016). Atmospheric Correction for Satellite Ocean Color Radiometry.
   NASA Technical Memorandum 2016-217551.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants


@dataclass
class SensorBand:
    """
    Spectral characteristics of a sensor band.

    Parameters
    ----------
    center_wavelength : float
        Nominal center wavelength in nm.
    fwhm_low : float
        Lower wavelength at half-maximum response in nm.
    fwhm_high : float
        Upper wavelength at half-maximum response in nm.
    oob_low : float
        Lower wavelength at 0.1% of maximum response in nm.
    oob_high : float
        Upper wavelength at 0.1% of maximum response in nm.
    """

    center_wavelength: float
    fwhm_low: float
    fwhm_high: float
    oob_low: float
    oob_high: float

    @property
    def fwhm(self) -> float:
        """Full width at half maximum in nm."""
        return self.fwhm_high - self.fwhm_low

    @property
    def in_band_range(self) -> Tuple[float, float]:
        """Wavelength range for in-band integration (0.1% threshold)."""
        return (self.oob_low, self.oob_high)


# Sensor band definitions with OOB boundaries
# Based on instrument characterization data
SENSOR_BANDS: Dict[str, Dict[int, SensorBand]] = {
    "seawifs": {
        412: SensorBand(412, 402, 422, 380, 440),
        443: SensorBand(443, 433, 453, 410, 470),
        490: SensorBand(490, 480, 500, 455, 520),
        510: SensorBand(510, 500, 520, 475, 540),
        555: SensorBand(555, 545, 565, 520, 590),
        670: SensorBand(670, 660, 680, 635, 705),
        765: SensorBand(765, 755, 775, 725, 805),
        865: SensorBand(865, 855, 875, 825, 905),
    },
    "modis_aqua": {
        412: SensorBand(412, 407, 417, 390, 435),
        443: SensorBand(443, 438, 448, 420, 465),
        488: SensorBand(488, 483, 493, 460, 515),
        531: SensorBand(531, 526, 536, 505, 555),
        547: SensorBand(547, 542, 552, 520, 570),
        667: SensorBand(667, 662, 672, 640, 695),
        678: SensorBand(678, 673, 683, 650, 705),
        748: SensorBand(748, 743, 753, 720, 775),
        869: SensorBand(869, 862, 877, 835, 905),
    },
    "viirs": {
        411: SensorBand(411, 402, 420, 380, 445),
        443: SensorBand(443, 436, 450, 415, 470),
        486: SensorBand(486, 478, 494, 455, 515),
        551: SensorBand(551, 543, 559, 520, 580),
        671: SensorBand(671, 662, 680, 635, 705),
        745: SensorBand(745, 739, 751, 715, 775),
        862: SensorBand(862, 846, 878, 820, 910),
    },
}


def gaussian_srf(
    wavelength: ArrayLike,
    center: float,
    fwhm: float,
) -> NDArray[np.floating]:
    """
    Generate a Gaussian spectral response function.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths at which to evaluate the SRF in nm.
    center : float
        Center wavelength in nm.
    fwhm : float
        Full width at half maximum in nm.

    Returns
    -------
    srf : ndarray
        Normalized spectral response function values.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    srf = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    return srf


def tophat_srf(
    wavelength: ArrayLike,
    center: float,
    width: float = 10.0,
) -> NDArray[np.floating]:
    """
    Generate an ideal "top hat" spectral response function.

    This represents a perfect sensor with uniform response over the
    nominal bandwidth and zero response outside.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths at which to evaluate the SRF in nm.
    center : float
        Center wavelength in nm.
    width : float, optional
        Full bandwidth in nm. Default is 10 nm.

    Returns
    -------
    srf : ndarray
        Ideal spectral response function (1 inside band, 0 outside).
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    half_width = width / 2.0
    srf = np.where(
        (wavelength >= center - half_width) & (wavelength <= center + half_width),
        1.0,
        0.0,
    )
    return srf


def band_averaged_radiance(
    wavelength: ArrayLike,
    radiance: ArrayLike,
    srf: ArrayLike,
) -> float:
    """
    Compute band-averaged radiance weighted by spectral response function.

    Implements Equation 10.1 from TM-2016-217551:

    .. math::

        L_i = \\frac{\\int L(\\lambda) SRF_i(\\lambda) d\\lambda}
                    {\\int SRF_i(\\lambda) d\\lambda}

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in nm.
    radiance : array_like
        Spectral radiance at each wavelength.
    srf : array_like
        Spectral response function at each wavelength.

    Returns
    -------
    L_band : float
        Band-averaged radiance.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    radiance = np.asarray(radiance, dtype=np.float64)
    srf = np.asarray(srf, dtype=np.float64)

    numerator = np.trapezoid(radiance * srf, wavelength)
    denominator = np.trapezoid(srf, wavelength)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_oob_fractions(
    wavelength: ArrayLike,
    radiance: ArrayLike,
    srf: ArrayLike,
    lambda_low: float,
    lambda_high: float,
) -> Tuple[float, float, float]:
    """
    Compute fractional contributions from in-band and out-of-band wavelengths.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in nm.
    radiance : array_like
        Spectral radiance at each wavelength.
    srf : array_like
        Full spectral response function at each wavelength.
    lambda_low : float
        Lower wavelength boundary for in-band region in nm.
    lambda_high : float
        Upper wavelength boundary for in-band region in nm.

    Returns
    -------
    f_inband : float
        Fraction of signal from in-band wavelengths.
    f_low : float
        Fraction from wavelengths below lambda_low.
    f_high : float
        Fraction from wavelengths above lambda_high.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    radiance = np.asarray(radiance, dtype=np.float64)
    srf = np.asarray(srf, dtype=np.float64)

    # Total signal
    total = np.trapezoid(radiance * srf, wavelength)

    if total == 0:
        return 1.0, 0.0, 0.0

    # In-band contribution
    mask_in = (wavelength >= lambda_low) & (wavelength <= lambda_high)
    in_band = np.trapezoid(
        np.where(mask_in, radiance * srf, 0.0), wavelength
    )

    # Low out-of-band
    mask_low = wavelength < lambda_low
    oob_low = np.trapezoid(
        np.where(mask_low, radiance * srf, 0.0), wavelength
    )

    # High out-of-band
    mask_high = wavelength > lambda_high
    oob_high = np.trapezoid(
        np.where(mask_high, radiance * srf, 0.0), wavelength
    )

    return float(in_band / total), float(oob_low / total), float(oob_high / total)


def case1_rrs_spectrum(
    wavelength: ArrayLike,
    chl: float,
) -> NDArray[np.floating]:
    """
    Generate Case 1 water Rrs spectrum for given chlorophyll concentration.

    This is a simplified bio-optical model based on Morel and Maritorena (2001),
    used to generate Rrs spectra for OOB correction factor computation.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in nm.
    chl : float
        Chlorophyll concentration in mg/m³.

    Returns
    -------
    rrs : ndarray
        Remote-sensing reflectance spectrum in sr⁻¹.

    Notes
    -----
    This is a simplified model for OOB correction purposes. For accurate
    Rrs modeling, use a full bio-optical model like that in HydroLight.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)

    # Reference wavelength
    lambda_ref = 550.0

    # Absorption by water (simplified)
    aw = 0.0145 * np.exp(0.014 * (wavelength - 440.0))

    # Absorption by phytoplankton (Bricaud et al. 1998 approximation)
    a_star = 0.06 * (chl ** 0.65) * np.exp(-0.014 * (wavelength - 440.0))
    aph = a_star * np.where(wavelength < 700, 1.0, 0.1)

    # Total absorption
    a_total = aw + aph

    # Backscattering by water
    bbw = 0.0029 * (lambda_ref / wavelength) ** 4.3

    # Backscattering by particles
    bbp = 0.002 * (chl ** 0.62) * (lambda_ref / wavelength)

    # Total backscattering
    bb_total = bbw + bbp

    # Gordon's formula for Rrs (simplified)
    u = bb_total / (a_total + bb_total)
    rrs = 0.089 * u + 0.125 * u ** 2

    # Apply water absorption cutoff at red/NIR
    rrs = np.where(wavelength > 700, rrs * np.exp(-(wavelength - 700) / 50), rrs)

    return rrs


def compute_oob_correction_ratio(
    wavelength: ArrayLike,
    rrs_spectrum: ArrayLike,
    srf_full: ArrayLike,
    center: float,
    ideal_width: float = 10.0,
) -> float:
    """
    Compute OOB correction ratio for a single band and Rrs spectrum.

    Implements Equation 10.2 from TM-2016-217551:

    .. math::

        r(\\lambda_i, Chl_j) = \\frac{R_{rs}^{10}(\\lambda_i, Chl_j)}
                                      {R_{rs}^{full}(\\lambda_i, Chl_j)}

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in nm.
    rrs_spectrum : array_like
        Rrs spectrum at each wavelength.
    srf_full : array_like
        Full sensor spectral response function.
    center : float
        Band center wavelength in nm.
    ideal_width : float, optional
        Width of ideal top-hat response in nm. Default is 10 nm.

    Returns
    -------
    r : float
        Correction ratio R_rs^ideal / R_rs^full.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    rrs_spectrum = np.asarray(rrs_spectrum, dtype=np.float64)
    srf_full = np.asarray(srf_full, dtype=np.float64)

    # Compute Rrs with full SRF
    rrs_full = band_averaged_radiance(wavelength, rrs_spectrum, srf_full)

    # Compute Rrs with ideal 10-nm top-hat
    srf_ideal = tophat_srf(wavelength, center, ideal_width)
    rrs_ideal = band_averaged_radiance(wavelength, rrs_spectrum, srf_ideal)

    if rrs_full == 0:
        return 1.0

    return rrs_ideal / rrs_full


class OOBCorrectionLUT:
    """
    Look-up table for out-of-band correction factors.

    The correction factors are precomputed as functions of the uncorrected
    Rrs(490)/Rrs(555) ratio, following the methodology in Section 10 of
    TM-2016-217551.

    Parameters
    ----------
    sensor : str
        Sensor name ('seawifs', 'modis_aqua', 'viirs').

    Attributes
    ----------
    sensor : str
        Sensor name.
    bands : dict
        Dictionary of SensorBand objects for each wavelength.
    correction_coeffs : dict
        Polynomial coefficients for correction factor vs band ratio.
    """

    def __init__(self, sensor: str = "seawifs"):
        self.sensor = sensor.lower()

        if self.sensor not in SENSOR_BANDS:
            raise ValueError(
                f"Unknown sensor: {sensor}. "
                f"Available: {list(SENSOR_BANDS.keys())}"
            )

        self.bands = SENSOR_BANDS[self.sensor]
        self.correction_coeffs = self._compute_correction_coefficients()

    def _compute_correction_coefficients(self) -> Dict[int, NDArray[np.floating]]:
        """
        Compute polynomial fit coefficients for each band.

        Returns
        -------
        coeffs : dict
            Dictionary mapping band wavelength to polynomial coefficients.
        """
        # Wavelength grid for integration
        wavelength = np.arange(350, 1000, 1.0)

        # Chlorophyll values spanning Case 1 range
        chl_values = np.array([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5,
                               0.7, 1.0, 1.5, 2.0, 3.0])

        coeffs = {}

        for band_nm, band_info in self.bands.items():
            ratios = []
            corrections = []

            # Generate synthetic SRF (Gaussian approximation)
            srf = gaussian_srf(wavelength, band_info.center_wavelength,
                               band_info.fwhm)

            for chl in chl_values:
                # Generate Rrs spectrum
                rrs = case1_rrs_spectrum(wavelength, chl)

                # Get band-averaged values for ratio calculation
                srf_490 = gaussian_srf(wavelength, 490, 10)
                srf_555 = gaussian_srf(wavelength, 555, 10)
                rrs_490 = band_averaged_radiance(wavelength, rrs, srf_490)
                rrs_555 = band_averaged_radiance(wavelength, rrs, srf_555)

                if rrs_555 > 0:
                    band_ratio = rrs_490 / rrs_555
                    ratios.append(band_ratio)

                    # Compute correction factor
                    r = compute_oob_correction_ratio(
                        wavelength, rrs, srf,
                        band_info.center_wavelength
                    )
                    corrections.append(r)

            # Fit polynomial to correction vs ratio
            if len(ratios) >= 3:
                ratios = np.array(ratios)
                corrections = np.array(corrections)
                # Quadratic fit
                coeffs[band_nm] = np.polyfit(ratios, corrections, 2)
            else:
                # Default to no correction
                coeffs[band_nm] = np.array([0.0, 0.0, 1.0])

        return coeffs

    def get_correction_factor(
        self,
        band_wavelength: int,
        rrs_ratio: float,
    ) -> float:
        """
        Get OOB correction factor for a band given the Rrs(490)/Rrs(555) ratio.

        Parameters
        ----------
        band_wavelength : int
            Nominal band wavelength in nm.
        rrs_ratio : float
            Ratio of Rrs(490)/Rrs(555) from uncorrected measurements.

        Returns
        -------
        r : float
            Correction factor to multiply uncorrected Rrs.
        """
        if band_wavelength not in self.correction_coeffs:
            return 1.0

        coeffs = self.correction_coeffs[band_wavelength]

        # Clip ratio to valid range (approximately 0.1 to 5.0 for Case 1)
        rrs_ratio_clipped = np.clip(rrs_ratio, 0.1, 5.0)

        return float(np.polyval(coeffs, rrs_ratio_clipped))

    def apply_correction(
        self,
        rrs: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Apply OOB correction to all bands.

        Parameters
        ----------
        rrs : dict
            Dictionary of uncorrected Rrs values keyed by band wavelength.

        Returns
        -------
        rrs_corrected : dict
            Dictionary of OOB-corrected Rrs values.
        """
        # Get the band ratio for correction lookup
        # Use nearest available bands to 490 and 555
        rrs_490 = self._get_nearest_band(rrs, 490)
        rrs_555 = self._get_nearest_band(rrs, 555)

        if rrs_555 is None or rrs_555 <= 0:
            # Can't compute ratio, return uncorrected
            return rrs.copy()

        if rrs_490 is None:
            band_ratio = 1.0  # Default ratio
        else:
            band_ratio = rrs_490 / rrs_555

        # Apply correction to each band
        rrs_corrected = {}
        for band_nm, rrs_value in rrs.items():
            correction = self.get_correction_factor(band_nm, band_ratio)
            rrs_corrected[band_nm] = rrs_value * correction

        return rrs_corrected

    def _get_nearest_band(
        self,
        rrs: Dict[int, float],
        target: int,
    ) -> Optional[float]:
        """Find Rrs value for band nearest to target wavelength."""
        if target in rrs:
            return rrs[target]

        # Find nearest
        bands = list(rrs.keys())
        if not bands:
            return None

        nearest = min(bands, key=lambda x: abs(x - target))
        if abs(nearest - target) < 30:  # Within 30 nm
            return rrs[nearest]

        return None


def apply_oob_correction(
    rrs: Dict[int, float],
    sensor: str = "seawifs",
) -> Dict[int, float]:
    """
    Apply out-of-band correction to Rrs measurements.

    This is a convenience function that creates an OOBCorrectionLUT and
    applies the correction in one step.

    Parameters
    ----------
    rrs : dict
        Dictionary of uncorrected Rrs values keyed by band wavelength in nm.
    sensor : str, optional
        Sensor name. Default is 'seawifs'.

    Returns
    -------
    rrs_corrected : dict
        Dictionary of OOB-corrected Rrs values.

    Examples
    --------
    >>> rrs = {412: 0.008, 443: 0.007, 490: 0.005, 555: 0.003, 670: 0.001}
    >>> rrs_corr = apply_oob_correction(rrs, sensor='seawifs')
    """
    lut = OOBCorrectionLUT(sensor)
    return lut.apply_correction(rrs)


def oob_correction_for_hyperspectral(
    wavelength: ArrayLike,
    rrs_hyperspectral: ArrayLike,
    sensor: str,
    band_wavelength: int,
) -> float:
    """
    Compute band-averaged Rrs from hyperspectral data using sensor SRF.

    When comparing satellite data with unfiltered hyperspectral in-situ
    measurements, the hyperspectral spectrum should be convolved with the
    satellite sensor SRF rather than applying OOB correction to satellite data.

    Parameters
    ----------
    wavelength : array_like
        Hyperspectral wavelengths in nm.
    rrs_hyperspectral : array_like
        Hyperspectral Rrs spectrum in sr⁻¹.
    sensor : str
        Sensor name.
    band_wavelength : int
        Nominal band wavelength in nm.

    Returns
    -------
    rrs_band : float
        Band-averaged Rrs using the sensor SRF.

    Notes
    -----
    Per Section 10 of TM-2016-217551:
    - For satellite vs in-situ multispectral: apply OOB to satellite
    - For satellite vs filtered hyperspectral: apply OOB to satellite
    - For satellite vs unfiltered hyperspectral: convolve hyperspectral with SRF
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    rrs_hyperspectral = np.asarray(rrs_hyperspectral, dtype=np.float64)

    if sensor.lower() not in SENSOR_BANDS:
        raise ValueError(f"Unknown sensor: {sensor}")

    bands = SENSOR_BANDS[sensor.lower()]
    if band_wavelength not in bands:
        raise ValueError(
            f"Band {band_wavelength} not found for sensor {sensor}"
        )

    band_info = bands[band_wavelength]

    # Generate SRF (Gaussian approximation)
    srf = gaussian_srf(wavelength, band_info.center_wavelength, band_info.fwhm)

    return band_averaged_radiance(wavelength, rrs_hyperspectral, srf)
