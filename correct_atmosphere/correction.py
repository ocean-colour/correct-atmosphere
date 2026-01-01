"""
Atmospheric Correction for Satellite Ocean Color Radiometry
============================================================

Main atmospheric correction class implementing the NASA OBPG algorithm
as documented in NASA TM-2016-217551 (Mobley et al., 2016).

This module provides the primary interface for performing atmospheric
correction on satellite ocean color data, converting top-of-atmosphere
(TOA) radiances to water-leaving radiances and remote-sensing reflectances.

References
----------
.. [1] Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
       Atmospheric Correction for Satellite Ocean Color Radiometry.
       NASA Technical Memorandum NASA/TM-2016-217551.

.. [2] Gordon, H.R. and Wang, M. (1994). Retrieval of water-leaving radiance
       and aerosol optical thickness over the oceans with SeaWiFS: a
       preliminary algorithm. Applied Optics, 33, 443-452.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np

from . import constants
from . import rayleigh
from . import gases
from . import glint
from . import whitecaps
from . import aerosols
from . import transmittance
from . import normalization


@dataclass
class GeometryAngles:
    """
    Solar and viewing geometry angles.

    All angles are in degrees.

    Attributes
    ----------
    solar_zenith : float or ndarray
        Solar zenith angle [degrees].
    solar_azimuth : float or ndarray
        Solar azimuth angle [degrees].
    view_zenith : float or ndarray
        Sensor viewing zenith angle [degrees].
    view_azimuth : float or ndarray
        Sensor viewing azimuth angle [degrees].
    relative_azimuth : float or ndarray, optional
        Relative azimuth angle between sun and sensor [degrees].
        Computed from solar_azimuth and view_azimuth if not provided.
    """

    solar_zenith: Union[float, np.ndarray]
    solar_azimuth: Union[float, np.ndarray]
    view_zenith: Union[float, np.ndarray]
    view_azimuth: Union[float, np.ndarray]
    relative_azimuth: Optional[Union[float, np.ndarray]] = None

    def __post_init__(self):
        """Compute relative azimuth if not provided."""
        if self.relative_azimuth is None:
            self.relative_azimuth = np.abs(self.view_azimuth - self.solar_azimuth)
            # Normalize to [0, 180]
            self.relative_azimuth = np.where(
                self.relative_azimuth > 180,
                360 - self.relative_azimuth,
                self.relative_azimuth,
            )

    @property
    def solar_zenith_rad(self) -> Union[float, np.ndarray]:
        """Solar zenith angle in radians."""
        return np.deg2rad(self.solar_zenith)

    @property
    def view_zenith_rad(self) -> Union[float, np.ndarray]:
        """View zenith angle in radians."""
        return np.deg2rad(self.view_zenith)

    @property
    def relative_azimuth_rad(self) -> Union[float, np.ndarray]:
        """Relative azimuth angle in radians."""
        return np.deg2rad(self.relative_azimuth)

    @property
    def air_mass_factor(self) -> Union[float, np.ndarray]:
        """
        Geometric air mass factor M = 1/cos(θs) + 1/cos(θv).

        See Equation 6.3 in NASA TM-2016-217551.
        """
        return rayleigh.geometric_air_mass_factor(
            self.solar_zenith, self.view_zenith
        )


@dataclass
class AncillaryData:
    """
    Ancillary data required for atmospheric correction.

    See Table 5.2 in NASA TM-2016-217551.

    Attributes
    ----------
    pressure : float or ndarray
        Sea level atmospheric pressure [hPa]. Default is 1013.25.
    wind_speed : float or ndarray
        Wind speed at 10m height [m/s]. Default is 5.0.
    ozone : float or ndarray
        Ozone column concentration [Dobson units]. Default is 300.
    water_vapor : float or ndarray
        Water vapor column [g/cm²]. Default is 1.5.
    no2_total : float or ndarray
        Total NO2 column concentration [molecules/cm²]. Default is 1e16.
    no2_stratospheric : float or ndarray
        Stratospheric NO2 (above 200m) [molecules/cm²]. Default is 1e16.
    relative_humidity : float or ndarray
        Relative humidity for aerosol models [%]. Default is 80.
    sea_surface_temperature : float or ndarray
        Sea surface temperature [°C]. Default is 20.
    sea_surface_salinity : float or ndarray
        Sea surface salinity [PSU]. Default is 35.
    """

    pressure: Union[float, np.ndarray] = 1013.25
    wind_speed: Union[float, np.ndarray] = 5.0
    ozone: Union[float, np.ndarray] = 300.0
    water_vapor: Union[float, np.ndarray] = 1.5
    no2_total: Union[float, np.ndarray] = 1.0e16
    no2_stratospheric: Union[float, np.ndarray] = 1.0e16
    relative_humidity: Union[float, np.ndarray] = 80.0
    sea_surface_temperature: Union[float, np.ndarray] = 20.0
    sea_surface_salinity: Union[float, np.ndarray] = 35.0


@dataclass
class CorrectionFlags:
    """
    Flags indicating correction status and quality.

    Attributes
    ----------
    glint_masked : bool or ndarray
        True if pixel was masked due to excessive sun glint.
    atmospheric_correction_warning : bool or ndarray
        True if atmospheric correction may be unreliable.
    negative_water_leaving : bool or ndarray
        True if negative water-leaving radiance was computed.
    iteration_limit_reached : bool or ndarray
        True if non-black-pixel iteration did not converge.
    turbid_water : bool or ndarray
        True if non-black-pixel correction was applied.
    straylight : bool or ndarray
        True if straylight contamination is suspected.
    cloud : bool or ndarray
        True if cloud contamination is detected.
    """

    glint_masked: Union[bool, np.ndarray] = False
    atmospheric_correction_warning: Union[bool, np.ndarray] = False
    negative_water_leaving: Union[bool, np.ndarray] = False
    iteration_limit_reached: Union[bool, np.ndarray] = False
    turbid_water: Union[bool, np.ndarray] = False
    straylight: Union[bool, np.ndarray] = False
    cloud: Union[bool, np.ndarray] = False


@dataclass
class CorrectionResult:
    """
    Results from atmospheric correction.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths of the bands [nm].
    rrs : ndarray
        Remote-sensing reflectance Rrs [sr⁻¹].
        This is the NASA OBPG standard product [ρw]ᵉˣ_N / π.
    nLw : ndarray
        Exact normalized water-leaving radiance [Lw]ᵉˣ_N
        [mW cm⁻² µm⁻¹ sr⁻¹].
    rho_w : ndarray
        Exact normalized water-leaving reflectance [ρw]ᵉˣ_N [dimensionless].
    Lw : ndarray
        Water-leaving radiance at sea surface [mW cm⁻² µm⁻¹ sr⁻¹].
    La : ndarray
        Aerosol path radiance at TOA [mW cm⁻² µm⁻¹ sr⁻¹].
    taua : ndarray
        Aerosol optical thickness at each wavelength.
    angstrom : float or ndarray
        Angstrom exponent for the retrieved aerosol.
    chlorophyll : float or ndarray
        Estimated chlorophyll concentration [mg/m³].
    flags : CorrectionFlags
        Quality and status flags.
    """

    wavelengths: np.ndarray
    rrs: np.ndarray
    nLw: np.ndarray
    rho_w: np.ndarray
    Lw: np.ndarray
    La: np.ndarray
    taua: np.ndarray
    angstrom: Union[float, np.ndarray]
    chlorophyll: Union[float, np.ndarray]
    flags: CorrectionFlags


class AtmosphericCorrection:
    """
    Atmospheric correction processor for ocean color satellite data.

    This class implements the NASA Ocean Biology Processing Group (OBPG)
    atmospheric correction algorithm as documented in NASA TM-2016-217551.
    The algorithm converts measured top-of-atmosphere (TOA) radiances to
    normalized water-leaving radiances and remote-sensing reflectances.

    Parameters
    ----------
    sensor : str
        Sensor name. One of 'seawifs', 'modis_aqua', 'modis_terra',
        'viirs_npp', 'viirs_noaa20'.
    apply_polarization : bool, optional
        Whether to apply polarization correction. Default is True.
    apply_brdf : bool, optional
        Whether to apply BRDF correction. Default is True.
    apply_outofband : bool, optional
        Whether to apply out-of-band correction. Default is True.
    glint_threshold : float, optional
        Normalized sun glint threshold for masking [sr⁻¹].
        Default is 0.005.
    max_iterations : int, optional
        Maximum iterations for non-black-pixel correction.
        Default is 10.
    convergence_threshold : float, optional
        Convergence threshold for iterative correction.
        Default is 0.02 (2%).

    Attributes
    ----------
    sensor : str
        The configured sensor name.
    wavelengths : ndarray
        Center wavelengths for the sensor bands [nm].
    nir_bands : tuple
        Indices of NIR bands used for aerosol correction.

    Examples
    --------
    >>> from correct_atmosphere import AtmosphericCorrection
    >>> ac = AtmosphericCorrection('modis_aqua')
    >>> result = ac.process(Lt, geometry, ancillary)
    >>> print(result.rrs)

    Notes
    -----
    The atmospheric correction process follows these steps (Figure 5.1):

    1. Correct for gaseous absorption (O3, NO2)
    2. Correct for polarization sensitivity
    3. Remove whitecap/foam contribution
    4. Remove Rayleigh scattering contribution
    5. Remove sun glint contribution
    6. Determine and remove aerosol contribution
    7. Normalize to [Lw]_N
    8. Iterate if non-black-pixel
    9. Apply bandpass correction
    10. Apply BRDF correction to get [Lw]ᵉˣ_N

    References
    ----------
    .. [1] Mobley et al. (2016). NASA/TM-2016-217551.
    """

    # Supported sensors
    SUPPORTED_SENSORS = {
        "seawifs",
        "modis_aqua",
        "modis_terra",
        "viirs_npp",
        "viirs_noaa20",
    }

    def __init__(
        self,
        sensor: str,
        apply_polarization: bool = True,
        apply_brdf: bool = True,
        apply_outofband: bool = True,
        glint_threshold: float = 0.005,
        max_iterations: int = 10,
        convergence_threshold: float = 0.02,
    ):
        sensor = sensor.lower()
        if sensor not in self.SUPPORTED_SENSORS:
            raise ValueError(
                f"Unsupported sensor '{sensor}'. "
                f"Supported sensors: {self.SUPPORTED_SENSORS}"
            )

        self.sensor = sensor
        self.apply_polarization = apply_polarization
        self.apply_brdf = apply_brdf
        self.apply_outofband = apply_outofband
        self.glint_threshold = glint_threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Get sensor-specific parameters
        self._setup_sensor()

        # Initialize LUTs (lazy loading)
        self._rayleigh_lut = None
        self._aerosol_lut = None
        self._brdf_lut = None

    def _setup_sensor(self):
        """Configure sensor-specific parameters."""
        sensor_key = self.sensor.upper()
        if sensor_key == "MODIS_AQUA":
            sensor_key = "MODIS"
        elif sensor_key == "MODIS_TERRA":
            sensor_key = "MODIS"
        elif sensor_key in ("VIIRS_NPP", "VIIRS_NOAA20"):
            sensor_key = "VIIRS"
        else:
            sensor_key = self.sensor.upper()

        if sensor_key in constants.SENSOR_BANDS:
            self.wavelengths = np.array(
                constants.SENSOR_BANDS[sensor_key]["center_wavelengths"]
            )
            self.bandwidths = np.array(
                constants.SENSOR_BANDS[sensor_key]["bandwidths"]
            )
        else:
            raise ValueError(f"No band definitions for sensor {self.sensor}")

        # NIR reference bands for aerosol correction
        if sensor_key in constants.NIR_REFERENCE_BANDS:
            self.nir_band_short = constants.NIR_REFERENCE_BANDS[sensor_key]["short"]
            self.nir_band_long = constants.NIR_REFERENCE_BANDS[sensor_key]["long"]
        else:
            # Default to last two bands
            self.nir_band_short = self.wavelengths[-2]
            self.nir_band_long = self.wavelengths[-1]

        # Find indices of NIR bands
        self.nir_idx_short = np.argmin(np.abs(self.wavelengths - self.nir_band_short))
        self.nir_idx_long = np.argmin(np.abs(self.wavelengths - self.nir_band_long))

    @property
    def rayleigh_lut(self) -> rayleigh.RayleighLUT:
        """Lazy-load Rayleigh lookup table."""
        if self._rayleigh_lut is None:
            self._rayleigh_lut = rayleigh.RayleighLUT(sensor=self.sensor)
        return self._rayleigh_lut

    @property
    def aerosol_lut(self) -> aerosols.AerosolLUT:
        """Lazy-load aerosol lookup table."""
        if self._aerosol_lut is None:
            self._aerosol_lut = aerosols.AerosolLUT()
        return self._aerosol_lut

    @property
    def brdf_lut(self) -> normalization.BRDFCorrection:
        """Lazy-load BRDF correction lookup table."""
        if self._brdf_lut is None:
            self._brdf_lut = normalization.BRDFCorrection()
        return self._brdf_lut

    def process(
        self,
        Lt: np.ndarray,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
        F0: Optional[np.ndarray] = None,
        day_of_year: int = 1,
    ) -> CorrectionResult:
        """
        Perform atmospheric correction on TOA radiances.

        Parameters
        ----------
        Lt : ndarray
            Top-of-atmosphere radiance [mW cm⁻² µm⁻¹ sr⁻¹].
            Shape: (n_bands,) or (n_bands, n_pixels) or (n_bands, ny, nx).
        geometry : GeometryAngles
            Solar and viewing geometry.
        ancillary : AncillaryData
            Ancillary environmental data.
        F0 : ndarray, optional
            Extraterrestrial solar irradiance at mean Earth-Sun distance
            [mW cm⁻² µm⁻¹]. If None, uses tabulated values.
        day_of_year : int, optional
            Day of year for Earth-Sun distance correction. Default is 1.

        Returns
        -------
        CorrectionResult
            Atmospheric correction results including Rrs, nLw, and flags.

        Notes
        -----
        The algorithm follows the flowchart in Figure 5.1 of NASA TM-2016-217551.
        """
        # Initialize flags
        flags = CorrectionFlags()

        # Get extraterrestrial solar irradiance if not provided
        if F0 is None:
            F0 = self._get_solar_irradiance()

        # Convert Lt to reflectance
        # ρt = π * Lt / (F0 * cos(θs))
        cos_sza = np.cos(geometry.solar_zenith_rad)
        rho_t = np.pi * Lt / (F0[:, np.newaxis] * cos_sza) if Lt.ndim > 1 else np.pi * Lt / (F0 * cos_sza)

        # Step 1: Correct for gaseous absorption (Section 6.2)
        rho_t_gas_corrected = self._correct_gases(
            rho_t, geometry, ancillary
        )

        # Step 2: Correct for polarization (Section 11)
        if self.apply_polarization:
            rho_t_pol_corrected = self._correct_polarization(
                rho_t_gas_corrected, geometry
            )
        else:
            rho_t_pol_corrected = rho_t_gas_corrected

        # Step 3: Remove whitecap contribution (Section 8)
        rho_wc = self._compute_whitecap_reflectance(ancillary.wind_speed)
        t_dv = self._compute_diffuse_transmittance(geometry, ancillary)

        # Step 4: Remove Rayleigh contribution (Section 6.1)
        rho_r = self._compute_rayleigh_reflectance(geometry, ancillary)

        # Step 5: Compute and mask sun glint (Section 7)
        L_GN = glint.normalized_sun_glint(
            geometry.solar_zenith,
            geometry.view_zenith,
            geometry.relative_azimuth,
            ancillary.wind_speed,
        )
        glint_mask = L_GN > self.glint_threshold
        flags.glint_masked = glint_mask

        rho_g = self._compute_glint_reflectance(
            geometry, ancillary, L_GN, F0
        )

        # Compute aerosol + water reflectance
        # ρAw = ρt - ρr - T*ρg - t*ρwc
        rho_Aw = rho_t_pol_corrected - rho_r - rho_g - t_dv * rho_wc

        # Step 6: Remove aerosol contribution (Section 9)
        # First try black-pixel assumption
        rho_A, aerosol_model, taua = self._compute_aerosol_reflectance(
            rho_Aw, geometry, ancillary
        )

        # Compute initial water-leaving reflectance
        rho_w = (rho_Aw - rho_A) / t_dv

        # Step 7-8: Normalize and iterate if needed
        chlorophyll = self._estimate_chlorophyll(rho_w)

        if aerosols.should_apply_nonblack_pixel(chlorophyll):
            flags.turbid_water = True
            rho_w, rho_A, taua, converged = self._iterate_nonblack_pixel(
                rho_t_pol_corrected,
                rho_r,
                rho_g,
                rho_wc,
                t_dv,
                geometry,
                ancillary,
            )
            if not converged:
                flags.iteration_limit_reached = True
                flags.atmospheric_correction_warning = True
            chlorophyll = self._estimate_chlorophyll(rho_w)

        # Check for negative values
        if np.any(rho_w < 0):
            flags.negative_water_leaving = True
            flags.atmospheric_correction_warning = True

        # Step 9: Apply out-of-band correction (Section 10)
        if self.apply_outofband:
            rho_w = self._correct_outofband(rho_w)

        # Step 10: Apply BRDF correction (Section 3.2)
        if self.apply_brdf:
            rho_w_ex = self._apply_brdf_correction(
                rho_w, geometry, ancillary.wind_speed, chlorophyll
            )
        else:
            rho_w_ex = rho_w

        # Convert to standard products
        # Rrs = [ρw]ᵉˣ_N / π (Eq. 3.10)
        rrs = rho_w_ex / np.pi

        # [Lw]ᵉˣ_N = [ρw]ᵉˣ_N * F0 / π (Eq. 3.8)
        nLw = rho_w_ex * F0 / np.pi if rho_w_ex.ndim == 1 else rho_w_ex * F0[:, np.newaxis] / np.pi

        # Water-leaving radiance at surface
        Lw = rho_w * F0 * cos_sza / np.pi if rho_w.ndim == 1 else rho_w * F0[:, np.newaxis] * cos_sza / np.pi

        # Aerosol path radiance
        La = rho_A * F0 * cos_sza / np.pi if rho_A.ndim == 1 else rho_A * F0[:, np.newaxis] * cos_sza / np.pi

        # Angstrom exponent
        angstrom = aerosols.angstrom_exponent(
            taua[self.nir_idx_short],
            taua[self.nir_idx_long],
            self.nir_band_short,
            self.nir_band_long,
        )

        return CorrectionResult(
            wavelengths=self.wavelengths,
            rrs=rrs,
            nLw=nLw,
            rho_w=rho_w_ex,
            Lw=Lw,
            La=La,
            taua=taua,
            angstrom=angstrom,
            chlorophyll=chlorophyll,
            flags=flags,
        )

    def _get_solar_irradiance(self) -> np.ndarray:
        """Get extraterrestrial solar irradiance for sensor bands."""
        # Placeholder - should load from sensor-specific tables
        # Typical values in mW cm⁻² µm⁻¹
        # Based on Thuillier 2003 solar spectrum
        f0_typical = {
            412: 172.912,
            443: 187.622,
            469: 196.243,
            488: 194.933,
            490: 194.933,
            510: 188.149,
            531: 185.747,
            547: 186.539,
            555: 183.869,
            645: 157.811,
            667: 152.255,
            670: 151.567,
            678: 148.052,
            748: 128.065,
            765: 123.478,
            859: 97.517,
            862: 96.867,
            865: 95.824,
            869: 95.299,
        }

        F0 = np.zeros(len(self.wavelengths))
        for i, wl in enumerate(self.wavelengths):
            # Find closest tabulated value
            closest = min(f0_typical.keys(), key=lambda x: abs(x - wl))
            F0[i] = f0_typical[closest]

        return F0

    def _correct_gases(
        self,
        rho_t: np.ndarray,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
    ) -> np.ndarray:
        """Apply gaseous absorption corrections (O3, NO2)."""
        # Ozone transmittance (Eq. 6.4)
        t_o3 = gases.ozone_transmittance(
            self.wavelengths,
            ancillary.ozone,
            geometry.solar_zenith,
            geometry.view_zenith,
            self.sensor,
        )

        # NO2 correction (Section 6.2.2)
        # For TOA reflectance, use stratospheric NO2
        t_no2 = gases.gas_transmittance(
            self.wavelengths,
            geometry.solar_zenith,
            geometry.view_zenith,
            o3_column=0,  # Already applied
            no2_column=ancillary.no2_stratospheric,
            sensor=self.sensor,
        )

        # Combined correction
        t_gas = t_o3 * t_no2

        # Correct reflectance
        if rho_t.ndim == 1:
            rho_corrected = rho_t / t_gas
        else:
            rho_corrected = rho_t / t_gas[:, np.newaxis]

        return rho_corrected

    def _correct_polarization(
        self,
        rho_t: np.ndarray,
        geometry: GeometryAngles,
    ) -> np.ndarray:
        """Apply polarization correction (Section 11)."""
        # Placeholder - requires sensor-specific Mueller matrix
        # and Rayleigh Stokes vector computation
        # For now, return uncorrected
        return rho_t

    def _compute_whitecap_reflectance(
        self,
        wind_speed: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute whitecap reflectance (Section 8)."""
        rho_wc = np.zeros(len(self.wavelengths))
        for i, wl in enumerate(self.wavelengths):
            rho_wc[i] = whitecaps.whitecap_reflectance(wind_speed, wl)
        return rho_wc

    def _compute_rayleigh_reflectance(
        self,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
    ) -> np.ndarray:
        """Compute Rayleigh reflectance (Section 6.1)."""
        rho_r = np.zeros(len(self.wavelengths))

        for i, wl in enumerate(self.wavelengths):
            # Get from LUT or compute
            rho_r_std = self.rayleigh_lut.interpolate(
                wl,
                geometry.solar_zenith,
                geometry.view_zenith,
                geometry.relative_azimuth,
                ancillary.wind_speed,
            )

            # Apply pressure correction (Eq. 6.2)
            tau_r = rayleigh.rayleigh_optical_thickness(wl, ancillary.pressure)
            tau_r_std = rayleigh.rayleigh_optical_thickness(wl)

            rho_r[i] = rayleigh.rayleigh_reflectance_pressure_corrected(
                rho_r_std,
                tau_r,
                tau_r_std,
                geometry.air_mass_factor,
                wl,
            )

        return rho_r

    def _compute_glint_reflectance(
        self,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
        L_GN: Union[float, np.ndarray],
        F0: np.ndarray,
    ) -> np.ndarray:
        """Compute sun glint reflectance (Section 7)."""
        rho_g = np.zeros(len(self.wavelengths))

        for i, wl in enumerate(self.wavelengths):
            # Two-path transmittance
            tau_total = rayleigh.rayleigh_optical_thickness(wl, ancillary.pressure)
            # Note: Should include aerosol optical thickness, but that's
            # determined iteratively
            T_two = glint.two_path_transmittance(
                geometry.solar_zenith,
                geometry.view_zenith,
                tau_total,
            )
            rho_g[i] = glint.sun_glint_reflectance(L_GN, T_two)

        return rho_g

    def _compute_diffuse_transmittance(
        self,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
    ) -> np.ndarray:
        """Compute diffuse transmittance in viewing direction."""
        t_dv = np.zeros(len(self.wavelengths))

        for i, wl in enumerate(self.wavelengths):
            tau_r = rayleigh.rayleigh_optical_thickness(wl, ancillary.pressure)
            t_dv[i] = transmittance.diffuse_transmittance(
                geometry.view_zenith,
                tau_r,
                tau_aerosol=0.1,  # Approximate; updated in iteration
            )

        return t_dv

    def _compute_aerosol_reflectance(
        self,
        rho_Aw: np.ndarray,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
    ) -> Tuple[np.ndarray, aerosols.AerosolModel, np.ndarray]:
        """Compute aerosol reflectance using black-pixel assumption."""
        # Extract NIR reflectances
        rho_Aw_nir = np.array([
            rho_Aw[self.nir_idx_short],
            rho_Aw[self.nir_idx_long],
        ])

        # Apply black-pixel correction
        result = aerosols.black_pixel_correction(
            rho_Aw_nir,
            np.array([self.nir_band_short, self.nir_band_long]),
            self.wavelengths,
            geometry.solar_zenith,
            geometry.view_zenith,
            geometry.relative_azimuth,
            ancillary.relative_humidity,
            self.aerosol_lut,
        )

        rho_A = result["rho_a"]
        model = result["aerosol_model"]

        # Compute AOT at each wavelength
        taua = np.zeros(len(self.wavelengths))
        taua_865 = result.get("taua_865", 0.1)  # Reference AOT

        for i, wl in enumerate(self.wavelengths):
            taua[i] = aerosols.aerosol_optical_thickness(
                wl,
                self.nir_band_long,
                taua_865,
                model.angstrom_exponent if model else 1.0,
            )

        return rho_A, model, taua

    def _iterate_nonblack_pixel(
        self,
        rho_t: np.ndarray,
        rho_r: np.ndarray,
        rho_g: np.ndarray,
        rho_wc: np.ndarray,
        t_dv: np.ndarray,
        geometry: GeometryAngles,
        ancillary: AncillaryData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Iterative non-black-pixel correction (Section 9.3).

        Returns
        -------
        rho_w : ndarray
            Water-leaving reflectance.
        rho_A : ndarray
            Aerosol reflectance.
        taua : ndarray
            Aerosol optical thickness.
        converged : bool
            Whether iteration converged.
        """
        # Initial black-pixel estimate
        rho_Aw = rho_t - rho_r - rho_g - t_dv * rho_wc
        rho_A, model, taua = self._compute_aerosol_reflectance(
            rho_Aw, geometry, ancillary
        )
        rho_w = (rho_Aw - rho_A) / t_dv

        rrs_nir_prev = np.array([0.0, 0.0])

        for iteration in range(self.max_iterations):
            # Estimate Rrs at NIR from visible bands (Bailey et al. 2010)
            rrs = rho_w / np.pi

            # Find indices for 443, 555, 670 nm bands
            idx_443 = np.argmin(np.abs(self.wavelengths - 443))
            idx_555 = np.argmin(np.abs(self.wavelengths - 555))
            idx_670 = np.argmin(np.abs(self.wavelengths - 670))

            rrs_nir = aerosols.estimate_nir_rrs(
                rrs[idx_443],
                rrs[idx_555],
                rrs[idx_670],
                self.nir_band_short,
                self.nir_band_long,
            )

            # Check convergence
            if iteration > 0:
                change = np.abs(rrs_nir - rrs_nir_prev) / (rrs_nir_prev + 1e-10)
                if np.all(change < self.convergence_threshold):
                    return rho_w, rho_A, taua, True

            rrs_nir_prev = rrs_nir.copy()

            # Update aerosol estimate
            rho_w_nir = rrs_nir * np.pi
            rho_Aw_corrected = rho_Aw.copy()
            rho_Aw_corrected[self.nir_idx_short] -= t_dv[self.nir_idx_short] * rho_w_nir[0]
            rho_Aw_corrected[self.nir_idx_long] -= t_dv[self.nir_idx_long] * rho_w_nir[1]

            # Recompute aerosol
            rho_A, model, taua = self._compute_aerosol_reflectance(
                rho_Aw_corrected, geometry, ancillary
            )
            rho_w = (rho_Aw - rho_A) / t_dv

        return rho_w, rho_A, taua, False

    def _estimate_chlorophyll(
        self,
        rho_w: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """Estimate chlorophyll concentration from reflectance."""
        # OC3/OC4 type algorithm
        # Find band indices
        idx_443 = np.argmin(np.abs(self.wavelengths - 443))
        idx_490 = np.argmin(np.abs(self.wavelengths - 490))
        idx_510 = np.argmin(np.abs(self.wavelengths - 510))
        idx_555 = np.argmin(np.abs(self.wavelengths - 555))

        # Maximum band ratio
        rrs = rho_w / np.pi
        ratios = np.array([
            rrs[idx_443] / rrs[idx_555],
            rrs[idx_490] / rrs[idx_555],
            rrs[idx_510] / rrs[idx_555],
        ])
        max_ratio = np.max(ratios, axis=0)

        # OC3M coefficients (approximate)
        log_ratio = np.log10(np.maximum(max_ratio, 0.01))
        a = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280]
        log_chl = a[0] + a[1] * log_ratio + a[2] * log_ratio**2 + a[3] * log_ratio**3 + a[4] * log_ratio**4

        return 10.0 ** log_chl

    def _correct_outofband(
        self,
        rho_w: np.ndarray,
    ) -> np.ndarray:
        """Apply out-of-band correction (Section 10)."""
        # Placeholder - requires sensor-specific correction factors
        return rho_w

    def _apply_brdf_correction(
        self,
        rho_w: np.ndarray,
        geometry: GeometryAngles,
        wind_speed: Union[float, np.ndarray],
        chlorophyll: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Apply BRDF correction (Section 3.2)."""
        rho_w_ex = np.zeros_like(rho_w)

        for i, wl in enumerate(self.wavelengths):
            if wl > 700:
                # No BRDF correction for NIR bands
                rho_w_ex[i] = rho_w[i]
            else:
                correction = self.brdf_lut.correction_factor(
                    wl,
                    geometry.solar_zenith,
                    geometry.view_zenith,
                    geometry.relative_azimuth,
                    chlorophyll,
                    wind_speed,
                )
                rho_w_ex[i] = rho_w[i] * correction

        return rho_w_ex
