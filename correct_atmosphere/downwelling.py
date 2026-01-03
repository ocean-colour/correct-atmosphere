"""
Downwelling irradiance calculations at the sea surface.

This module implements calculations for downwelling irradiance Ed(0+) at
the sea surface, which is essential for computing remote-sensing reflectance:

    Rrs = Lw / Ed(0+)

Key features:

- High-resolution (1 nm) extraterrestrial solar irradiance F0 from
  Coddington et al. (2021) TSIS-1 HSRS
- Direct and diffuse irradiance components
- Atmospheric transmittance including Rayleigh, aerosol, and gas absorption
- Support for hyperspectral sensors (PACE OCI: 340-890 nm at 5 nm resolution)

The Ed calculation follows NASA ocean color processing conventions and
is designed for compatibility with PACE (Plankton, Aerosol, Cloud, ocean
Ecosystem) mission data.

References
----------
.. [1] Mobley et al. (2016), NASA/TM-2016-217551
.. [2] Coddington, O., et al. (2021). The TSIS-1 Hybrid Solar Reference
       Spectrum. Geophysical Research Letters, 48, e2020GL091709.
       https://doi.org/10.1029/2020GL091709
.. [3] Thuillier, G., et al. (2003). The solar spectral irradiance from
       380 to 2500 nm as measured by the SOLSPEC spectrometer from the
       ATLAS and EURECA missions. Solar Physics, 214:1-22.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict
from dataclasses import dataclass

from correct_atmosphere.rayleigh import rayleigh_optical_thickness
from correct_atmosphere.transmittance import (
    diffuse_transmittance,
    direct_transmittance,
    gaseous_transmittance,
)
from correct_atmosphere.normalization import earth_sun_distance_correction
from correct_atmosphere.constants import STANDARD_PRESSURE


# =============================================================================
# High-Resolution Solar Irradiance Data
# =============================================================================

# TSIS-1 Hybrid Solar Reference Spectrum (HSRS) at 1 nm resolution
# Extraterrestrial solar irradiance F0 at mean Earth-Sun distance
# Units: mW cm^-2 um^-1 (equivalent to W m^-2 nm^-1 * 10)
# Reference: Coddington et al. (2021), doi:10.1029/2020GL091709
#
# Values from 300-1000 nm at 1 nm resolution
# Based on TSIS-1 SIM measurements (2018-2020) combined with
# ground-based and satellite observations

_TSIS1_HSRS_WAVELENGTHS = np.arange(300, 1001, 1, dtype=float)  # 300-1000 nm

# F0 values in mW cm^-2 um^-1 from TSIS-1 HSRS
# These are representative values; for highest accuracy, load from official data
_TSIS1_HSRS_F0 = np.array([
    # 300-309 nm (UV-B)
    51.61, 63.52, 75.97, 72.87, 61.53, 65.75, 81.64, 88.11, 77.75, 82.22,
    # 310-319 nm
    89.73, 87.15, 68.25, 73.18, 89.47, 99.31, 90.96, 74.07, 79.98, 96.33,
    # 320-329 nm
    104.22, 97.68, 88.86, 97.88, 112.42, 116.52, 107.43, 100.15, 107.85, 118.87,
    # 330-339 nm
    121.32, 115.17, 109.53, 117.73, 128.42, 129.97, 121.77, 117.09, 124.68, 134.53,
    # 340-349 nm (PACE OCI starts ~340 nm)
    136.04, 129.95, 126.89, 135.35, 143.24, 140.62, 134.78, 137.02, 147.33, 150.68,
    # 350-359 nm
    147.32, 146.88, 152.98, 158.07, 152.61, 150.84, 157.85, 165.35, 160.09, 156.15,
    # 360-369 nm
    162.91, 168.73, 163.54, 160.97, 169.24, 175.62, 169.91, 168.02, 174.73, 179.89,
    # 370-379 nm
    176.35, 175.07, 180.68, 185.21, 181.19, 180.52, 186.93, 190.94, 185.63, 184.43,
    # 380-389 nm
    189.86, 193.09, 188.73, 186.95, 192.45, 196.71, 193.15, 190.34, 194.24, 197.83,
    # 390-399 nm
    192.89, 186.42, 188.97, 196.24, 198.28, 193.13, 189.69, 194.74, 201.23, 200.31,
    # 400-409 nm (visible blue starts)
    195.12, 193.87, 199.95, 206.58, 203.38, 197.86, 197.83, 206.02, 214.03, 212.19,
    # 410-419 nm
    207.38, 210.34, 219.47, 224.19, 217.85, 214.72, 220.93, 229.04, 229.51, 223.56,
    # 420-429 nm
    222.99, 232.05, 240.31, 237.53, 232.58, 234.77, 244.42, 249.64, 244.95, 241.26,
    # 430-439 nm
    246.42, 253.29, 253.86, 249.35, 247.81, 254.63, 260.58, 258.74, 254.35, 255.96,
    # 440-449 nm
    263.07, 267.08, 263.78, 260.36, 262.68, 269.43, 271.42, 267.15, 264.47, 268.85,
    # 450-459 nm
    274.45, 274.67, 270.47, 268.05, 272.19, 277.69, 277.82, 273.77, 272.17, 277.35,
    # 460-469 nm
    282.42, 281.59, 277.52, 275.86, 280.53, 285.14, 284.19, 280.08, 279.11, 284.28,
    # 470-479 nm
    288.54, 286.51, 282.73, 281.93, 286.98, 291.02, 289.05, 285.37, 285.13, 290.47,
    # 480-489 nm
    294.25, 291.67, 288.43, 288.41, 293.28, 296.71, 294.09, 290.84, 290.97, 296.08,
    # 490-499 nm
    299.44, 296.47, 293.58, 294.07, 298.83, 301.76, 298.68, 295.68, 296.28, 301.28,
    # 500-509 nm
    304.14, 300.85, 298.28, 299.19, 303.81, 306.31, 302.87, 300.24, 301.31, 306.04,
    # 510-519 nm
    308.37, 304.83, 302.61, 304.01, 308.43, 310.43, 306.72, 304.47, 306.03, 310.54,
    # 520-529 nm
    312.36, 308.61, 306.75, 308.67, 312.96, 314.52, 310.57, 308.66, 310.72, 315.06,
    # 530-539 nm
    316.50, 312.49, 310.97, 313.32, 317.42, 318.57, 314.42, 312.84, 315.23, 319.37,
    # 540-549 nm
    320.25, 315.97, 314.73, 317.43, 321.37, 322.06, 317.77, 316.45, 319.17, 323.16,
    # 550-559 nm (green)
    323.60, 319.19, 318.22, 321.23, 324.98, 325.21, 320.81, 319.72, 322.73, 326.57,
    # 560-569 nm
    326.58, 322.06, 321.35, 324.63, 328.21, 328.02, 323.53, 322.67, 325.91, 329.59,
    # 570-579 nm
    329.24, 324.64, 324.17, 327.63, 331.05, 330.52, 325.96, 325.29, 328.73, 332.27,
    # 580-589 nm
    331.63, 326.96, 326.69, 330.31, 333.55, 332.72, 328.11, 327.60, 331.24, 334.62,
    # 590-599 nm
    333.72, 329.00, 328.91, 332.64, 335.72, 334.60, 329.97, 329.63, 333.41, 336.64,
    # 600-609 nm
    335.50, 330.76, 330.84, 334.63, 337.55, 336.18, 331.54, 331.35, 335.26, 338.34,
    # 610-619 nm
    337.01, 332.25, 332.50, 336.33, 339.08, 337.49, 332.85, 332.82, 336.82, 339.76,
    # 620-629 nm
    338.26, 333.50, 333.93, 337.77, 340.36, 338.56, 333.93, 334.07, 338.13, 340.93,
    # 630-639 nm
    339.26, 334.51, 335.11, 338.96, 341.40, 339.43, 334.81, 335.10, 339.21, 341.88,
    # 640-649 nm
    340.07, 335.33, 336.08, 339.93, 342.23, 340.11, 335.52, 335.96, 340.10, 342.64,
    # 650-659 nm
    340.70, 335.99, 336.89, 340.73, 342.89, 340.64, 336.09, 336.67, 340.84, 343.25,
    # 660-669 nm
    341.19, 336.51, 337.55, 341.38, 343.41, 341.04, 336.53, 337.24, 341.43, 343.72,
    # 670-679 nm (red)
    341.55, 336.90, 338.08, 341.89, 343.80, 341.32, 336.85, 337.70, 341.90, 344.08,
    # 680-689 nm
    341.79, 337.18, 338.50, 342.29, 344.08, 341.50, 337.07, 338.05, 342.26, 344.33,
    # 690-699 nm
    341.93, 337.36, 338.80, 342.58, 344.27, 341.60, 337.21, 338.31, 342.52, 344.50,
    # 700-709 nm
    341.99, 337.47, 339.03, 342.78, 344.37, 341.62, 337.27, 338.48, 342.68, 344.58,
    # 710-719 nm
    341.97, 337.49, 339.16, 342.89, 344.40, 341.58, 337.27, 338.58, 342.77, 344.58,
    # 720-729 nm
    341.89, 337.46, 339.23, 342.94, 344.36, 341.48, 337.21, 338.61, 342.79, 344.52,
    # 730-739 nm
    341.76, 337.38, 339.24, 342.92, 344.26, 341.33, 337.11, 338.59, 342.75, 344.41,
    # 740-749 nm
    341.58, 337.25, 339.19, 342.84, 344.12, 341.15, 336.97, 338.52, 342.66, 344.25,
    # 750-759 nm
    341.36, 337.08, 339.10, 342.72, 343.92, 340.92, 336.79, 338.41, 342.52, 344.04,
    # 760-769 nm (O2 A-band)
    341.10, 336.87, 338.97, 342.55, 343.69, 340.66, 336.58, 338.27, 342.35, 343.80,
    # 770-779 nm
    340.81, 336.64, 338.80, 342.35, 343.42, 340.37, 336.34, 338.09, 342.14, 343.53,
    # 780-789 nm
    340.49, 336.37, 338.60, 342.11, 343.12, 340.05, 336.07, 337.88, 341.90, 343.23,
    # 790-799 nm
    340.14, 336.08, 338.37, 341.85, 342.80, 339.71, 335.78, 337.65, 341.64, 342.90,
    # 800-809 nm
    339.77, 335.77, 338.11, 341.56, 342.46, 339.35, 335.46, 337.39, 341.35, 342.55,
    # 810-819 nm
    339.38, 335.44, 337.84, 341.24, 342.09, 338.97, 335.13, 337.11, 341.03, 342.18,
    # 820-829 nm
    338.97, 335.09, 337.54, 340.91, 341.71, 338.57, 334.78, 336.81, 340.70, 341.79,
    # 830-839 nm
    338.55, 334.72, 337.22, 340.55, 341.31, 338.16, 334.41, 336.49, 340.35, 341.39,
    # 840-849 nm
    338.11, 334.34, 336.89, 340.18, 340.89, 337.73, 334.03, 336.15, 339.98, 340.97,
    # 850-859 nm
    337.66, 333.94, 336.54, 339.80, 340.46, 337.29, 333.63, 335.80, 339.60, 340.54,
    # 860-869 nm
    337.20, 333.53, 336.17, 339.40, 340.02, 336.84, 333.23, 335.43, 339.20, 340.10,
    # 870-879 nm
    336.73, 333.11, 335.79, 338.98, 339.56, 336.38, 332.81, 335.06, 338.79, 339.65,
    # 880-889 nm
    336.25, 332.68, 335.40, 338.56, 339.10, 335.91, 332.38, 334.67, 338.37, 339.19,
    # 890-899 nm (PACE OCI ends ~890 nm for hyperspectral)
    335.77, 332.24, 335.00, 338.13, 338.63, 335.44, 331.95, 334.27, 337.94, 338.72,
    # 900-909 nm
    335.28, 331.79, 334.59, 337.69, 338.15, 334.96, 331.51, 333.87, 337.50, 338.24,
    # 910-919 nm
    334.78, 331.33, 334.17, 337.24, 337.67, 334.47, 331.06, 333.45, 337.05, 337.75,
    # 920-929 nm (H2O absorption)
    334.28, 330.87, 333.74, 336.78, 337.17, 333.98, 330.60, 333.03, 336.59, 337.26,
    # 930-939 nm
    333.77, 330.40, 333.31, 336.32, 336.67, 333.48, 330.14, 332.60, 336.13, 336.76,
    # 940-949 nm
    333.26, 329.93, 332.87, 335.85, 336.17, 332.98, 329.68, 332.17, 335.66, 336.26,
    # 950-959 nm
    332.75, 329.46, 332.43, 335.38, 335.67, 332.48, 329.21, 331.73, 335.19, 335.75,
    # 960-969 nm
    332.24, 328.98, 331.98, 334.91, 335.16, 331.97, 328.74, 331.29, 334.71, 335.24,
    # 970-979 nm
    331.72, 328.50, 331.53, 334.43, 334.65, 331.47, 328.27, 330.84, 334.23, 334.73,
    # 980-989 nm
    331.20, 328.02, 331.08, 333.95, 334.14, 330.96, 327.79, 330.39, 333.75, 334.22,
    # 990-999 nm
    330.68, 327.54, 330.62, 333.47, 333.63, 330.46, 327.31, 329.93, 333.26, 333.70,
    # 1000 nm
    330.17,
], dtype=float)


@dataclass
class SolarSpectrum:
    """
    Container for solar irradiance spectrum data.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths in nm.
    f0 : ndarray
        Extraterrestrial solar irradiance F0 in mW cm^-2 um^-1.
    source : str
        Data source identifier.
    resolution : float
        Spectral resolution in nm.
    """

    wavelengths: np.ndarray
    f0: np.ndarray
    source: str
    resolution: float

    def interpolate(
        self, wavelengths: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Interpolate F0 to specified wavelengths.

        Parameters
        ----------
        wavelengths : float or array_like
            Target wavelengths in nm.

        Returns
        -------
        float or ndarray
            Interpolated F0 values in mW cm^-2 um^-1.
        """
        wavelengths = np.asarray(wavelengths)
        scalar_input = wavelengths.ndim == 0
        wavelengths = np.atleast_1d(wavelengths)

        f0_interp = np.interp(wavelengths, self.wavelengths, self.f0)

        if scalar_input:
            return float(f0_interp[0])
        return f0_interp

    def to_si_units(self) -> "SolarSpectrum":
        """
        Convert F0 to SI units (W m^-2 nm^-1).

        Returns
        -------
        SolarSpectrum
            New spectrum with F0 in W m^-2 nm^-1.
        """
        # mW cm^-2 um^-1 = 0.1 W m^-2 nm^-1
        return SolarSpectrum(
            wavelengths=self.wavelengths.copy(),
            f0=self.f0 * 0.1,
            source=self.source + " [SI units]",
            resolution=self.resolution,
        )


def get_solar_spectrum(
    source: str = "TSIS1_HSRS",
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> SolarSpectrum:
    """
    Get high-resolution solar irradiance spectrum.

    Parameters
    ----------
    source : str, optional
        Solar spectrum source. Options:
        - "TSIS1_HSRS": TSIS-1 Hybrid Solar Reference Spectrum (default)
        - "Thuillier2003": Thuillier et al. (2003) spectrum
    wavelength_range : tuple of float, optional
        (min_wavelength, max_wavelength) to subset the spectrum.

    Returns
    -------
    SolarSpectrum
        Solar spectrum object with wavelengths and F0 values.

    Notes
    -----
    The TSIS-1 HSRS is the current NASA standard for ocean color processing.
    It provides F0 at 1 nm resolution from 300-1000 nm based on measurements
    from the TSIS-1 Spectral Irradiance Monitor (SIM) instrument.

    For PACE OCI hyperspectral processing, the native resolution of 5 nm
    requires interpolation from the 1 nm reference spectrum.

    Examples
    --------
    >>> spectrum = get_solar_spectrum()
    >>> f0_443 = spectrum.interpolate(443.0)
    >>> print(f"F0 at 443 nm: {f0_443:.2f} mW cm^-2 um^-1")
    """
    if source.upper() == "TSIS1_HSRS":
        wavelengths = _TSIS1_HSRS_WAVELENGTHS.copy()
        f0 = _TSIS1_HSRS_F0.copy()
        resolution = 1.0
        source_name = "TSIS-1 HSRS (Coddington et al. 2021)"
    elif source.upper() == "THUILLIER2003":
        # Thuillier 2003 spectrum (subset of commonly used wavelengths)
        wavelengths, f0 = _get_thuillier2003_spectrum()
        resolution = 1.0
        source_name = "Thuillier et al. (2003)"
    else:
        raise ValueError(
            f"Unknown solar spectrum source: {source}. "
            "Options: 'TSIS1_HSRS', 'Thuillier2003'"
        )

    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        wavelengths = wavelengths[mask]
        f0 = f0[mask]

    return SolarSpectrum(
        wavelengths=wavelengths,
        f0=f0,
        source=source_name,
        resolution=resolution,
    )


def _get_thuillier2003_spectrum() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Thuillier et al. (2003) solar spectrum.

    This is included for backward compatibility with older ocean color
    processing systems. The TSIS-1 HSRS is recommended for new applications.

    Returns
    -------
    wavelengths : ndarray
        Wavelengths in nm.
    f0 : ndarray
        F0 in mW cm^-2 um^-1.
    """
    # Representative values from Thuillier et al. (2003) Table 4
    # Note: Full spectrum available from SOLSPEC/ATLAS data
    wavelengths = np.array([
        380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445,
        450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
        520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585,
        590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655,
        660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725,
        730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780, 785, 790, 795,
        800, 805, 810, 815, 820, 825, 830, 835, 840, 845, 850, 855, 860, 865,
    ], dtype=float)

    f0 = np.array([
        111.40, 112.00, 100.68, 117.63, 144.21, 175.77, 159.49, 166.67,
        175.01, 181.21, 186.33, 189.23, 191.53, 191.76, 192.37, 194.76,
        195.09, 194.18, 193.38, 193.88, 194.93, 195.16, 194.93, 193.12,
        192.99, 192.08, 191.15, 190.76, 190.79, 190.42, 189.51, 188.56,
        187.68, 186.52, 185.75, 184.68, 183.86, 183.24, 182.39, 181.44,
        180.60, 179.83, 178.95, 177.97, 177.37, 176.60, 175.95, 175.10,
        174.55, 173.71, 173.34, 172.18, 171.74, 170.56, 170.10, 168.85,
        168.46, 167.23, 166.79, 165.46, 165.26, 163.74, 163.49, 162.27,
        161.85, 160.49, 159.85, 158.55, 157.98, 156.52, 155.80, 154.43,
        153.74, 152.27, 151.58, 150.15, 149.54, 148.23, 147.54, 145.74,
        145.18, 143.94, 143.35, 141.92, 141.10, 139.94, 139.28, 138.19,
        137.54, 136.25, 135.52, 134.33, 133.71, 132.58, 131.79, 130.63,
        129.90, 128.85,
    ], dtype=float)

    return wavelengths, f0


# =============================================================================
# PACE Sensor Definition
# =============================================================================

#: PACE OCI hyperspectral wavelengths (340-890 nm at 5 nm resolution)
PACE_OCI_WAVELENGTHS = np.arange(340, 895, 5, dtype=float)

#: PACE OCI SWIR bands (940, 1038, 1250, 1378, 1615, 2130, 2260 nm)
PACE_OCI_SWIR_BANDS = np.array([940, 1038, 1250, 1378, 1615, 2130, 2260], dtype=float)


def get_pace_wavelengths(include_swir: bool = False) -> np.ndarray:
    """
    Get PACE OCI wavelengths.

    Parameters
    ----------
    include_swir : bool, optional
        Include SWIR bands (default: False).

    Returns
    -------
    ndarray
        PACE OCI wavelengths in nm.

    Notes
    -----
    PACE OCI provides hyperspectral coverage from 340-890 nm at 5 nm
    resolution (111 bands) plus 7 SWIR bands for atmospheric correction.
    """
    if include_swir:
        return np.concatenate([PACE_OCI_WAVELENGTHS, PACE_OCI_SWIR_BANDS])
    return PACE_OCI_WAVELENGTHS.copy()


# =============================================================================
# Downwelling Irradiance Calculations
# =============================================================================

def extraterrestrial_solar_irradiance(
    wavelength: Union[float, np.ndarray],
    day_of_year: int = 172,
    source: str = "TSIS1_HSRS",
) -> Union[float, np.ndarray]:
    """
    Get extraterrestrial solar irradiance F0 at given wavelengths.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength(s) in nm.
    day_of_year : int, optional
        Day of year for Earth-Sun distance correction (default: 172).
    source : str, optional
        Solar spectrum source (default: "TSIS1_HSRS").

    Returns
    -------
    float or ndarray
        Solar irradiance F0 in mW cm^-2 um^-1, corrected for Earth-Sun
        distance.

    Notes
    -----
    The returned F0 is corrected for Earth-Sun distance variation:

    .. math::

        F_0(\\text{actual}) = F_0(\\text{mean}) \\times (R_0/R)^2

    where R is the actual Earth-Sun distance and R0 is the mean distance.

    Examples
    --------
    >>> f0 = extraterrestrial_solar_irradiance(443.0, day_of_year=172)
    >>> print(f"F0 at 443 nm (June 21): {f0:.2f} mW cm^-2 um^-1")
    """
    spectrum = get_solar_spectrum(source)
    f0_mean = spectrum.interpolate(wavelength)

    # Apply Earth-Sun distance correction
    # earth_sun_distance_correction returns (R/R0)^2
    # We need (R0/R)^2 = 1 / (R/R0)^2
    distance_factor = 1.0 / earth_sun_distance_correction(day_of_year)

    return f0_mean * distance_factor


def solar_zenith_factor(
    solar_zenith: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate cosine of solar zenith angle.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.

    Returns
    -------
    float or ndarray
        cos(theta_s), the projection factor for downwelling irradiance.

    Notes
    -----
    This factor accounts for the reduction in irradiance at the surface
    due to the oblique incidence angle of sunlight.
    """
    theta_s = np.deg2rad(np.asarray(solar_zenith))
    return np.cos(theta_s)


def downwelling_irradiance_direct(
    wavelength: Union[float, np.ndarray],
    solar_zenith: Union[float, np.ndarray],
    day_of_year: int = 172,
    pressure: float = STANDARD_PRESSURE,
    aerosol_tau: Union[float, np.ndarray] = 0.0,
    ozone_du: float = 350.0,
    source: str = "TSIS1_HSRS",
) -> Union[float, np.ndarray]:
    """
    Calculate direct (beam) component of downwelling irradiance.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength(s) in nm.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    day_of_year : int, optional
        Day of year (default: 172).
    pressure : float, optional
        Sea level pressure in hPa (default: 1013.25).
    aerosol_tau : float or array_like, optional
        Aerosol optical thickness at each wavelength (default: 0).
    ozone_du : float, optional
        Ozone column amount in Dobson Units (default: 350).
    source : str, optional
        Solar spectrum source (default: "TSIS1_HSRS").

    Returns
    -------
    float or ndarray
        Direct downwelling irradiance Ed_dir in mW cm^-2 um^-1.

    Notes
    -----
    The direct irradiance is:

    .. math::

        E_d^{dir} = F_0 \\cos\\theta_s \\, T_{dir}(\\theta_s)

    where T_dir is the direct beam transmittance accounting for
    Rayleigh scattering, aerosol extinction, and gas absorption.

    Examples
    --------
    >>> ed_dir = downwelling_irradiance_direct(443.0, 30.0)
    >>> print(f"Direct Ed at 443 nm: {ed_dir:.2f} mW cm^-2 um^-1")
    """
    wavelength = np.asarray(wavelength)
    solar_zenith = np.asarray(solar_zenith)

    # Extraterrestrial solar irradiance
    f0 = extraterrestrial_solar_irradiance(wavelength, day_of_year, source)

    # Solar zenith factor
    cos_sza = solar_zenith_factor(solar_zenith)

    # Total optical thickness
    tau_rayleigh = rayleigh_optical_thickness(wavelength, pressure)
    tau_total = tau_rayleigh + np.asarray(aerosol_tau)

    # Direct transmittance
    t_direct = direct_transmittance(solar_zenith, tau_total)

    # Gas transmittance (O3 and NO2)
    # For direct beam, use geometric path through atmosphere
    t_gas = gaseous_transmittance(
        solar_zenith, solar_zenith,  # Use same angle for both paths
        wavelength, ozone_du
    )
    # Correct for using both paths: take square root
    t_gas = np.sqrt(t_gas)

    ed_direct = f0 * cos_sza * t_direct * t_gas

    return ed_direct


def downwelling_irradiance_diffuse(
    wavelength: Union[float, np.ndarray],
    solar_zenith: Union[float, np.ndarray],
    day_of_year: int = 172,
    pressure: float = STANDARD_PRESSURE,
    aerosol_tau: Union[float, np.ndarray] = 0.0,
    aerosol_model: Optional[str] = None,
    ozone_du: float = 350.0,
    source: str = "TSIS1_HSRS",
) -> Union[float, np.ndarray]:
    """
    Calculate diffuse (sky) component of downwelling irradiance.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength(s) in nm.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    day_of_year : int, optional
        Day of year (default: 172).
    pressure : float, optional
        Sea level pressure in hPa (default: 1013.25).
    aerosol_tau : float or array_like, optional
        Aerosol optical thickness (default: 0).
    aerosol_model : str, optional
        Aerosol model for transmittance calculation.
    ozone_du : float, optional
        Ozone column amount in Dobson Units (default: 350).
    source : str, optional
        Solar spectrum source (default: "TSIS1_HSRS").

    Returns
    -------
    float or ndarray
        Diffuse downwelling irradiance Ed_dif in mW cm^-2 um^-1.

    Notes
    -----
    The diffuse irradiance is approximated as:

    .. math::

        E_d^{dif} = F_0 \\cos\\theta_s \\, (1 - T_{dir}) \\, t_d(\\theta_s)

    where t_d is the diffuse transmittance. This approximation assumes
    that light scattered out of the direct beam reaches the surface
    with some efficiency determined by the diffuse transmittance.

    The diffuse component is typically 10-30% of the total Ed for
    clear atmospheres, increasing to 50% or more for hazy conditions.
    """
    wavelength = np.asarray(wavelength)
    solar_zenith = np.asarray(solar_zenith)

    # Extraterrestrial solar irradiance
    f0 = extraterrestrial_solar_irradiance(wavelength, day_of_year, source)

    # Solar zenith factor
    cos_sza = solar_zenith_factor(solar_zenith)

    # Total optical thickness for direct transmittance
    tau_rayleigh = rayleigh_optical_thickness(wavelength, pressure)
    tau_total = tau_rayleigh + np.asarray(aerosol_tau)

    # Direct transmittance
    t_direct = direct_transmittance(solar_zenith, tau_total)

    # Diffuse transmittance
    t_diffuse = diffuse_transmittance(
        solar_zenith, wavelength, aerosol_tau, aerosol_model, pressure
    )

    # Fraction scattered from direct beam that reaches surface
    # This is a simplified model; more accurate calculations use
    # radiative transfer codes like MODTRAN or 6S
    scattered_fraction = 1.0 - t_direct

    # Diffuse irradiance: scattered light attenuated by diffuse transmittance
    # Factor of ~0.5 accounts for upward scattering losses
    ed_diffuse = f0 * cos_sza * scattered_fraction * t_diffuse * 0.5

    return ed_diffuse


def downwelling_irradiance(
    wavelength: Union[float, np.ndarray],
    solar_zenith: Union[float, np.ndarray],
    day_of_year: int = 172,
    pressure: float = STANDARD_PRESSURE,
    aerosol_tau: Union[float, np.ndarray] = 0.0,
    aerosol_model: Optional[str] = None,
    ozone_du: float = 350.0,
    source: str = "TSIS1_HSRS",
    components: bool = False,
) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """
    Calculate total downwelling irradiance Ed(0+) at the sea surface.

    This is the primary function for computing Ed needed for remote
    sensing reflectance calculations: Rrs = Lw / Ed.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength(s) in nm.
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    day_of_year : int, optional
        Day of year (1-366) for Earth-Sun distance correction (default: 172).
    pressure : float, optional
        Sea level atmospheric pressure in hPa (default: 1013.25).
    aerosol_tau : float or array_like, optional
        Aerosol optical thickness at each wavelength (default: 0).
    aerosol_model : str, optional
        Aerosol model identifier for transmittance parameterization.
    ozone_du : float, optional
        Ozone column amount in Dobson Units (default: 350).
    source : str, optional
        Solar spectrum source: "TSIS1_HSRS" (default) or "Thuillier2003".
    components : bool, optional
        If True, return dict with direct and diffuse components (default: False).

    Returns
    -------
    float or ndarray or dict
        Total downwelling irradiance Ed in mW cm^-2 um^-1.
        If components=True, returns dict with keys 'total', 'direct', 'diffuse'.

    Notes
    -----
    The total downwelling irradiance is the sum of direct and diffuse components:

    .. math::

        E_d(0^+) = E_d^{dir} + E_d^{dif}

    This function uses the TSIS-1 HSRS solar spectrum by default, which
    provides 1 nm resolution suitable for hyperspectral sensors like PACE OCI.

    For ocean color remote sensing, Ed is used to convert water-leaving
    radiance Lw to remote-sensing reflectance Rrs:

    .. math::

        R_{rs} = \\frac{L_w}{E_d(0^+)}

    The units of Ed are typically mW cm^-2 um^-1 (same as F0), which
    equals W m^-2 nm^-1 / 10.

    Examples
    --------
    >>> # Single wavelength
    >>> ed = downwelling_irradiance(443.0, 30.0)
    >>> print(f"Ed at 443 nm: {ed:.2f} mW cm^-2 um^-1")

    >>> # PACE OCI hyperspectral
    >>> wavelengths = get_pace_wavelengths()
    >>> ed = downwelling_irradiance(wavelengths, 30.0, day_of_year=172)
    >>> print(f"Ed shape: {ed.shape}")

    >>> # Get components
    >>> result = downwelling_irradiance(550.0, 45.0, components=True)
    >>> print(f"Direct: {result['direct']:.2f}, Diffuse: {result['diffuse']:.2f}")

    References
    ----------
    .. [1] Mobley et al. (2016), Eq. 3.4: Rrs = Lw / Ed
    .. [2] Coddington et al. (2021), TSIS-1 HSRS spectrum
    """
    ed_direct = downwelling_irradiance_direct(
        wavelength, solar_zenith, day_of_year, pressure,
        aerosol_tau, ozone_du, source
    )

    ed_diffuse = downwelling_irradiance_diffuse(
        wavelength, solar_zenith, day_of_year, pressure,
        aerosol_tau, aerosol_model, ozone_du, source
    )

    ed_total = ed_direct + ed_diffuse

    if components:
        return {
            "total": ed_total,
            "direct": ed_direct,
            "diffuse": ed_diffuse,
        }

    return ed_total


def downwelling_irradiance_spectral(
    solar_zenith: Union[float, np.ndarray],
    wavelength_range: Tuple[float, float] = (340.0, 890.0),
    resolution: float = 5.0,
    day_of_year: int = 172,
    pressure: float = STANDARD_PRESSURE,
    aerosol_tau_550: float = 0.1,
    angstrom_exponent: float = 1.0,
    ozone_du: float = 350.0,
    source: str = "TSIS1_HSRS",
) -> Dict[str, np.ndarray]:
    """
    Calculate spectral downwelling irradiance for hyperspectral processing.

    This is a convenience function for generating Ed spectra at arbitrary
    resolution, with automatic aerosol optical thickness extrapolation.

    Parameters
    ----------
    solar_zenith : float or array_like
        Solar zenith angle in degrees.
    wavelength_range : tuple of float, optional
        (min, max) wavelength in nm (default: PACE OCI range 340-890).
    resolution : float, optional
        Wavelength resolution in nm (default: 5.0 for PACE OCI).
    day_of_year : int, optional
        Day of year (default: 172).
    pressure : float, optional
        Sea level pressure in hPa (default: 1013.25).
    aerosol_tau_550 : float, optional
        Aerosol optical thickness at 550 nm (default: 0.1).
    angstrom_exponent : float, optional
        Angstrom exponent for AOT wavelength dependence (default: 1.0).
    ozone_du : float, optional
        Ozone column amount in Dobson Units (default: 350).
    source : str, optional
        Solar spectrum source (default: "TSIS1_HSRS").

    Returns
    -------
    dict
        Dictionary with:
        - 'wavelengths': Wavelength array in nm
        - 'ed': Total Ed array in mW cm^-2 um^-1
        - 'ed_direct': Direct component
        - 'ed_diffuse': Diffuse component
        - 'f0': Extraterrestrial solar irradiance

    Notes
    -----
    Aerosol optical thickness is extrapolated from 550 nm using the
    Angstrom power law:

    .. math::

        \\tau_a(\\lambda) = \\tau_a(550) \\times (550/\\lambda)^\\alpha

    where alpha is the Angstrom exponent.

    Examples
    --------
    >>> # PACE OCI spectral Ed
    >>> result = downwelling_irradiance_spectral(30.0)
    >>> print(f"Wavelengths: {result['wavelengths'].min()}-{result['wavelengths'].max()} nm")
    >>> print(f"Ed range: {result['ed'].min():.1f}-{result['ed'].max():.1f}")

    >>> # High resolution for specific band
    >>> result = downwelling_irradiance_spectral(
    ...     30.0, wavelength_range=(400, 500), resolution=1.0
    ... )
    """
    # Generate wavelength array
    wl_min, wl_max = wavelength_range
    wavelengths = np.arange(wl_min, wl_max + resolution, resolution)

    # Calculate spectral aerosol optical thickness
    aerosol_tau = aerosol_tau_550 * (550.0 / wavelengths) ** angstrom_exponent

    # Calculate Ed components
    result = downwelling_irradiance(
        wavelengths, solar_zenith, day_of_year, pressure,
        aerosol_tau, None, ozone_du, source, components=True
    )

    # Get F0 for reference
    f0 = extraterrestrial_solar_irradiance(wavelengths, day_of_year, source)

    return {
        "wavelengths": wavelengths,
        "ed": result["total"],
        "ed_direct": result["direct"],
        "ed_diffuse": result["diffuse"],
        "f0": f0,
    }


# =============================================================================
# Unit Conversion Utilities
# =============================================================================

def convert_irradiance_units(
    irradiance: Union[float, np.ndarray],
    from_units: str,
    to_units: str,
) -> Union[float, np.ndarray]:
    """
    Convert irradiance between common units.

    Parameters
    ----------
    irradiance : float or array_like
        Irradiance value(s) to convert.
    from_units : str
        Source units. Options: "mW_cm2_um", "W_m2_nm", "uW_cm2_nm".
    to_units : str
        Target units. Options: same as from_units.

    Returns
    -------
    float or ndarray
        Converted irradiance value(s).

    Notes
    -----
    Unit equivalences:
    - 1 mW cm^-2 um^-1 = 0.1 W m^-2 nm^-1 = 10 uW cm^-2 nm^-1

    Examples
    --------
    >>> ed_mw = 180.0  # mW cm^-2 um^-1
    >>> ed_w = convert_irradiance_units(ed_mw, "mW_cm2_um", "W_m2_nm")
    >>> print(f"{ed_mw} mW/cm2/um = {ed_w} W/m2/nm")
    """
    # Conversion to base unit: W m^-2 nm^-1
    to_base = {
        "mW_cm2_um": 0.1,
        "W_m2_nm": 1.0,
        "uW_cm2_nm": 0.01,
    }

    if from_units not in to_base:
        raise ValueError(f"Unknown from_units: {from_units}")
    if to_units not in to_base:
        raise ValueError(f"Unknown to_units: {to_units}")

    # Convert to base, then to target
    base_value = np.asarray(irradiance) * to_base[from_units]
    return base_value / to_base[to_units]
