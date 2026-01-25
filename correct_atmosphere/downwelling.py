"""
Downwelling irradiance calculations at the sea surface.

This module implements calculations for downwelling irradiance Ed(0+) at
the sea surface, which is essential for computing remote-sensing reflectance:

    Rrs = Lw / Ed(0+)

Key features:

- High-resolution (0.1 nm) extraterrestrial solar irradiance F0 from the
  TSIS-1 Hybrid Solar Reference Spectrum (HSRS) v2 (Coddington et al. 2021)
- Solar spectrum data loaded from official NetCDF file (202-2730 nm range)
- Includes measurement uncertainty from the TSIS-1 SIM instrument
- Direct and diffuse irradiance components
- Atmospheric transmittance including Rayleigh, aerosol, and gas absorption
- Support for hyperspectral sensors (PACE OCI: 340-890 nm at 5 nm resolution)

The Ed calculation follows NASA ocean color processing conventions and
is designed for compatibility with PACE (Plankton, Aerosol, Cloud, ocean
Ecosystem) mission data.

Data Requirements
-----------------
The TSIS-1 HSRS data file must be present at:
    correct_atmosphere/data/Solar/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc

This file can be downloaded from:
    https://lasp.colorado.edu/lisird/data/tsis1_hsrs

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
from pathlib import Path
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
# Data File Paths
# =============================================================================

#: Path to the TSIS-1 HSRS data file
_DATA_DIR = Path(__file__).parent / "data" / "Solar"
_TSIS1_HSRS_FILE = _DATA_DIR / "hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc"

# Cache for loaded solar spectrum data
_SOLAR_SPECTRUM_CACHE: Dict[str, "SolarSpectrum"] = {}


# =============================================================================
# Solar Spectrum Data Loading
# =============================================================================

@dataclass
class SolarSpectrum:
    """
    Container for solar irradiance spectrum data.

    This class holds extraterrestrial solar irradiance (F0) data and provides
    methods for interpolation and unit conversion. When loaded from TSIS-1 HSRS,
    measurement uncertainty is also available.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths in nm (vacuum wavelengths for TSIS-1 HSRS).
    f0 : ndarray
        Extraterrestrial solar irradiance F0 in mW cm^-2 um^-1.
        This is the mean solar distance value; use Earth-Sun distance
        correction for actual values on a specific day.
    source : str
        Data source identifier (e.g., "TSIS-1 HSRS v2 (Coddington et al. 2021)").
    resolution : float
        Nominal spectral resolution in nm.
    uncertainty : ndarray, optional
        Measurement uncertainty in F0 (same units as F0).
        Available for TSIS-1 HSRS data.

    Examples
    --------
    >>> spectrum = get_solar_spectrum(wavelength_range=(400, 700))
    >>> f0_550 = spectrum.interpolate(550.0)
    >>> print(f"F0 at 550 nm: {f0_550:.2f} mW cm^-2 um^-1")

    >>> # Convert to SI units
    >>> spectrum_si = spectrum.to_si_units()
    >>> print(f"F0 at 550 nm: {spectrum_si.interpolate(550.0):.4f} W m^-2 nm^-1")
    """

    wavelengths: np.ndarray
    f0: np.ndarray
    source: str
    resolution: float
    uncertainty: Optional[np.ndarray] = None

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
        unc = self.uncertainty * 0.1 if self.uncertainty is not None else None
        return SolarSpectrum(
            wavelengths=self.wavelengths.copy(),
            f0=self.f0 * 0.1,
            source=self.source + " [SI units]",
            resolution=self.resolution,
            uncertainty=unc,
        )


def _load_tsis1_hsrs(
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> SolarSpectrum:
    """
    Load TSIS-1 Hybrid Solar Reference Spectrum from NetCDF file.

    Loads the official TSIS-1 HSRS v2 data from the NetCDF file distributed
    by LASP (Laboratory for Atmospheric and Space Physics). The spectrum
    covers 202-2730 nm at 0.1 nm resolution (25,281 data points).

    The data includes:
    - Solar Spectral Irradiance (SSI) in W m^-2 nm^-1
    - Measurement uncertainty (SSI_UNC)
    - Vacuum wavelengths

    Values are converted to ocean color standard units (mW cm^-2 um^-1).

    Parameters
    ----------
    wavelength_range : tuple of float, optional
        (min_wavelength, max_wavelength) in nm to subset the spectrum.
        If None, the full 202-2730 nm range is returned.

    Returns
    -------
    SolarSpectrum
        Solar spectrum object containing:
        - wavelengths: array of wavelengths in nm
        - f0: extraterrestrial solar irradiance in mW cm^-2 um^-1
        - uncertainty: measurement uncertainty in mW cm^-2 um^-1
        - source: data provenance string
        - resolution: nominal spectral resolution (1.0 nm)

    Raises
    ------
    FileNotFoundError
        If the TSIS-1 HSRS data file is not found at the expected location.

    Notes
    -----
    Typical F0 values at ocean color wavelengths (mW cm^-2 um^-1):
    - 412 nm: ~187
    - 443 nm: ~197
    - 490 nm: ~208
    - 555 nm: ~193
    - 670 nm: ~153
    - 865 nm: ~96
    """
    import xarray as xr

    if not _TSIS1_HSRS_FILE.exists():
        raise FileNotFoundError(
            f"TSIS-1 HSRS data file not found: {_TSIS1_HSRS_FILE}\n"
            "Please download the file from: "
            "https://lasp.colorado.edu/lisird/data/tsis1_hsrs"
        )

    # Load the NetCDF file
    ds = xr.open_dataset(_TSIS1_HSRS_FILE)

    # Extract data
    # File uses "Vacuum Wavelength" and "SSI" (Solar Spectral Irradiance)
    wavelengths = ds["Vacuum Wavelength"].values
    ssi = ds["SSI"].values  # W m^-2 nm^-1
    ssi_unc = ds["SSI_UNC"].values  # W m^-2 nm^-1

    ds.close()

    # Convert from W m^-2 nm^-1 to mW cm^-2 um^-1
    # 1 W = 10^3 mW, 1 m^-2 = 10^-4 cm^-2, 1 nm^-1 = 10^3 um^-1
    # So: 1 W m^-2 nm^-1 = 10^3 * 10^-4 * 10^3 = 100 mW cm^-2 um^-1
    f0 = ssi * 100.0
    uncertainty = ssi_unc * 100.0

    # Apply wavelength range filter if specified
    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        wavelengths = wavelengths[mask]
        f0 = f0[mask]
        uncertainty = uncertainty[mask]

    return SolarSpectrum(
        wavelengths=wavelengths,
        f0=f0,
        source="TSIS-1 HSRS v2 (Coddington et al. 2021)",
        resolution=1.0,
        uncertainty=uncertainty,
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
        - "TSIS1_HSRS": TSIS-1 Hybrid Solar Reference Spectrum v2 (default)
          Loaded from NetCDF file, covers 202-2730 nm at 0.1 nm resolution.
        - "Thuillier2003": Thuillier et al. (2003) spectrum
          Hardcoded values, covers 380-875 nm at 5 nm resolution.
    wavelength_range : tuple of float, optional
        (min_wavelength, max_wavelength) in nm to subset the spectrum.
        If None, returns the full wavelength range of the source.

    Returns
    -------
    SolarSpectrum
        Solar spectrum object with wavelengths, F0 values (mW cm^-2 um^-1),
        and uncertainty (for TSIS-1 HSRS). Results are cached for efficiency.

    Notes
    -----
    The TSIS-1 HSRS v2 is the current NASA standard for ocean color processing.
    It provides F0 at 0.1 nm resolution from 202-2730 nm based on measurements
    from the TSIS-1 Spectral Irradiance Monitor (SIM) instrument, combined with
    ground-based and model data for wavelengths outside the SIM range.

    The TSIS-1 HSRS data file must be present at:
        correct_atmosphere/data/Solar/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc

    Download from: https://lasp.colorado.edu/lisird/data/tsis1_hsrs

    For PACE OCI hyperspectral processing (340-890 nm at 5 nm resolution),
    use the ``interpolate()`` method to resample to sensor wavelengths.

    Examples
    --------
    >>> spectrum = get_solar_spectrum()
    >>> f0_443 = spectrum.interpolate(443.0)
    >>> print(f"F0 at 443 nm: {f0_443:.2f} mW cm^-2 um^-1")
    F0 at 443 nm: 197.17 mW cm^-2 um^-1

    >>> # Get spectrum for visible range only
    >>> vis_spectrum = get_solar_spectrum(wavelength_range=(400, 700))
    >>> print(f"Visible range: {vis_spectrum.wavelengths[0]}-{vis_spectrum.wavelengths[-1]} nm")
    """
    # Create cache key
    cache_key = f"{source}_{wavelength_range}"

    # Check cache
    if cache_key in _SOLAR_SPECTRUM_CACHE:
        return _SOLAR_SPECTRUM_CACHE[cache_key]

    if source.upper() == "TSIS1_HSRS":
        spectrum = _load_tsis1_hsrs(wavelength_range)
    elif source.upper() == "THUILLIER2003":
        wavelengths, f0 = _get_thuillier2003_spectrum()

        if wavelength_range is not None:
            wl_min, wl_max = wavelength_range
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            wavelengths = wavelengths[mask]
            f0 = f0[mask]

        spectrum = SolarSpectrum(
            wavelengths=wavelengths,
            f0=f0,
            source="Thuillier et al. (2003)",
            resolution=5.0,  # Thuillier data is at 5 nm resolution
        )
    else:
        raise ValueError(
            f"Unknown solar spectrum source: {source}. "
            "Options: 'TSIS1_HSRS', 'Thuillier2003'"
        )

    # Cache the result
    _SOLAR_SPECTRUM_CACHE[cache_key] = spectrum

    return spectrum


def clear_solar_spectrum_cache() -> None:
    """Clear the solar spectrum cache to free memory."""
    _SOLAR_SPECTRUM_CACHE.clear()


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
