"""
Polarization Correction
=======================

Correction for sensor polarization sensitivity as described in
Section 11 of NASA TM-2016-217551.

Many satellite sensors have significant polarization sensitivity,
meaning the measured radiance depends on the state of polarization
of the incident light and the orientation of the sensor relative
to the plane of polarization. This module provides corrections for
these effects.

Theory
------
The state of polarization is described by the Stokes vector [I, Q, U, V]ᵀ,
where:

- I is the total radiance
- Q specifies linear polarization parallel/perpendicular to reference plane
- U specifies linear polarization at ±45° to reference plane
- V specifies circular polarization (usually negligible)

The sensor measures:

    Im = M · R(α) · It

where M is the sensor Mueller matrix, R(α) is a rotation matrix for angle α
between the meridional plane and sensor reference frame, and It is the TOA
Stokes vector.

References
----------
.. [1] Mobley et al. (2016). NASA/TM-2016-217551, Section 11.

.. [2] Gordon, H.R., Du, T., and Zhang, T. (1997). Atmospheric correction
       of ocean color sensors: analysis of the effects of residual instrument
       polarization sensitivity. Applied Optics, 36, 6938-6948.

.. [3] Meister, G., et al. (2005). Moderate-resolution imaging spectroradiometer
       ocean color polarization correction. Applied Optics, 44, 5524-5535.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class StokesVector:
    """
    Stokes vector representation of polarized radiance.

    Attributes
    ----------
    I : float or ndarray
        Total radiance (intensity).
    Q : float or ndarray
        Linear polarization component (parallel - perpendicular).
    U : float or ndarray
        Linear polarization component (±45°).
    V : float or ndarray
        Circular polarization component. Usually negligible.
    """

    I: Union[float, np.ndarray]
    Q: Union[float, np.ndarray]
    U: Union[float, np.ndarray]
    V: Union[float, np.ndarray] = 0.0

    def as_array(self) -> np.ndarray:
        """Return Stokes vector as numpy array."""
        return np.array([self.I, self.Q, self.U, self.V])

    @property
    def degree_of_polarization(self) -> Union[float, np.ndarray]:
        """
        Total degree of polarization.

        Returns
        -------
        float or ndarray
            Degree of polarization, 0 to 1.
        """
        return np.sqrt(self.Q**2 + self.U**2 + self.V**2) / (self.I + 1e-10)

    @property
    def degree_of_linear_polarization(self) -> Union[float, np.ndarray]:
        """
        Degree of linear polarization.

        Returns
        -------
        float or ndarray
            Linear polarization fraction, 0 to 1.
        """
        return np.sqrt(self.Q**2 + self.U**2) / (self.I + 1e-10)

    @property
    def q(self) -> Union[float, np.ndarray]:
        """Reduced Stokes parameter q = Q/I."""
        return self.Q / (self.I + 1e-10)

    @property
    def u(self) -> Union[float, np.ndarray]:
        """Reduced Stokes parameter u = U/I."""
        return self.U / (self.I + 1e-10)


def compute_rotation_angle(
    theta_v: Union[float, np.ndarray],
    phi_v: Union[float, np.ndarray],
    sensor_orientation: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute angle α between meridional plane and sensor reference.

    Simplified interface for computing the rotation angle.

    Parameters
    ----------
    theta_v : float or ndarray
        Sensor view zenith angle [degrees].
    phi_v : float or ndarray
        Sensor view azimuth angle [degrees].
    sensor_orientation : float or ndarray, optional
        Sensor roll angle relative to orbital plane [degrees].
        Default is 0.

    Returns
    -------
    float or ndarray
        Angle α between meridional plane and sensor reference [degrees].
    """
    # Simplified calculation assuming nadir-pointing sensor
    alpha = phi_v + sensor_orientation
    return alpha


def degree_of_polarization(stokes: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute degree of polarization from Stokes vector.

    Parameters
    ----------
    stokes : ndarray
        Stokes vector [I, Q, U, V].

    Returns
    -------
    float or ndarray
        Degree of polarization, 0 to 1.
    """
    I, Q, U, V = stokes[0], stokes[1], stokes[2], stokes[3]
    return np.sqrt(Q**2 + U**2 + V**2) / (I + 1e-10)


def rotation_matrix(alpha: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute Stokes vector rotation matrix R(α).

    Rotates the reference plane for Q and U components by angle α.
    See Equation 11.1 in NASA TM-2016-217551.

    Parameters
    ----------
    alpha : float or ndarray
        Rotation angle [degrees]. Positive for clockwise rotation
        as seen looking into the beam.

    Returns
    -------
    ndarray
        4×4 rotation matrix.

    Notes
    -----
    The rotation matrix is:

    .. math::

        R(\\alpha) = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(2\\alpha) & \\sin(2\\alpha) & 0 \\\\
            0 & -\\sin(2\\alpha) & \\cos(2\\alpha) & 0 \\\\
            0 & 0 & 0 & 1
        \\end{bmatrix}
    """
    alpha_rad = np.deg2rad(alpha)
    c2a = np.cos(2 * alpha_rad)
    s2a = np.sin(2 * alpha_rad)

    R = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c2a, s2a, 0.0],
        [0.0, -s2a, c2a, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return R


def rotate_stokes_vector(
    stokes: StokesVector,
    alpha: Union[float, np.ndarray],
) -> StokesVector:
    """
    Rotate Stokes vector reference frame by angle α.

    Parameters
    ----------
    stokes : StokesVector
        Input Stokes vector.
    alpha : float or ndarray
        Rotation angle [degrees].

    Returns
    -------
    StokesVector
        Rotated Stokes vector.
    """
    R = rotation_matrix(alpha)
    s_in = stokes.as_array()
    s_out = R @ s_in

    return StokesVector(
        I=s_out[0],
        Q=s_out[1],
        U=s_out[2],
        V=s_out[3],
    )


@dataclass
class MuellerMatrix:
    """
    Mueller matrix representation of sensor polarization sensitivity.

    Attributes
    ----------
    M11 : float
        First diagonal element (total intensity response).
    M12 : float
        Sensitivity to Q polarization.
    M13 : float
        Sensitivity to U polarization.
    M14 : float
        Sensitivity to V polarization (usually 0).
    """

    M11: float = 1.0
    M12: float = 0.0
    M13: float = 0.0
    M14: float = 0.0

    @property
    def matrix(self) -> np.ndarray:
        """Return full 4x4 Mueller matrix."""
        return np.array([
            [self.M11, self.M12, self.M13, self.M14],
            [self.M12, self.M11, 0.0, 0.0],
            [self.M13, 0.0, self.M11, 0.0],
            [self.M14, 0.0, 0.0, self.M11],
        ])

    @property
    def m12(self) -> float:
        """Reduced Mueller matrix element M12/M11."""
        return self.M12 / self.M11

    @property
    def m13(self) -> float:
        """Reduced Mueller matrix element M13/M11."""
        return self.M13 / self.M11

    @classmethod
    def ideal_sensor(cls) -> "MuellerMatrix":
        """Return Mueller matrix for ideal sensor (no polarization sensitivity)."""
        return cls(M11=1.0, M12=0.0, M13=0.0, M14=0.0)

    @classmethod
    def modis_aqua(cls, band: int) -> "MuellerMatrix":
        """
        Return Mueller matrix for MODIS Aqua at given band.

        Parameters
        ----------
        band : int
            Wavelength in nm (e.g., 412, 443, 490, etc.)

        Returns
        -------
        MuellerMatrix
            Approximate Mueller matrix for the band.
        """
        # Approximate values from Meister et al. 2005
        m12_values = {
            412: 0.02, 443: 0.02, 469: 0.01, 488: 0.01, 490: 0.01,
            531: 0.01, 547: 0.01, 555: 0.01, 645: 0.02, 667: 0.02,
            670: 0.02, 678: 0.02, 748: 0.03, 859: 0.04, 865: 0.04, 869: 0.04,
        }
        m13_values = {
            412: 0.01, 443: 0.01, 469: 0.005, 488: 0.005, 490: 0.005,
            531: 0.005, 547: 0.005, 555: 0.005, 645: 0.01, 667: 0.01,
            670: 0.01, 678: 0.01, 748: 0.01, 859: 0.02, 865: 0.02, 869: 0.02,
        }
        m12 = m12_values.get(band, 0.02)
        m13 = m13_values.get(band, 0.01)
        return cls(M11=1.0, M12=m12, M13=m13, M14=0.0)


def stokes_vector_rayleigh(
    theta_s: Union[float, np.ndarray],
    theta_v: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    tau_r: float,
    wavelength: float,
) -> np.ndarray:
    """
    Compute Rayleigh-scattered Stokes vector at TOA.

    Parameters
    ----------
    theta_s : float or ndarray
        Solar zenith angle [degrees].
    theta_v : float or ndarray
        View zenith angle [degrees].
    phi : float or ndarray
        Relative azimuth angle [degrees].
    tau_r : float
        Rayleigh optical thickness.
    wavelength : float
        Wavelength [nm] (for reference, not used in calculation).

    Returns
    -------
    ndarray
        Stokes vector [I, Q, U, V].
    """
    # Convert angles to radians
    theta_s_rad = np.deg2rad(theta_s)
    theta_v_rad = np.deg2rad(theta_v)
    phi_rad = np.deg2rad(phi)

    # Scattering angle
    cos_scatter = -np.cos(theta_s_rad) * np.cos(theta_v_rad) + \
                  np.sin(theta_s_rad) * np.sin(theta_v_rad) * np.cos(phi_rad)

    # Rayleigh phase function (unpolarized component)
    P_unpol = 0.75 * (1 + cos_scatter**2)

    # Single-scattering approximation for I
    I_r = tau_r * P_unpol / (4 * np.cos(theta_v_rad) + 1e-10)

    # Degree of polarization for Rayleigh scattering
    sin2_scatter = 1 - cos_scatter**2
    dolp = sin2_scatter / (1 + cos_scatter**2 + 1e-10)

    # Q and U components depend on scattering geometry
    Q_r = -I_r * dolp * np.cos(2 * phi_rad)
    U_r = -I_r * dolp * np.sin(2 * phi_rad)

    return np.array([I_r, Q_r, U_r, 0.0])


def compute_polarization_correction(
    M: MuellerMatrix,
    I_t: np.ndarray,
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute polarization correction factor.

    Parameters
    ----------
    M : MuellerMatrix
        Sensor Mueller matrix.
    I_t : ndarray
        TOA Stokes vector [I, Q, U, V].
    alpha : float or ndarray
        Rotation angle [degrees].

    Returns
    -------
    float or ndarray
        Polarization correction factor pc = Im/It.
    """
    m12 = M.m12
    m13 = M.m13

    alpha_rad = np.deg2rad(alpha)
    c2a = np.cos(2 * alpha_rad)
    s2a = np.sin(2 * alpha_rad)

    # Reduced Stokes parameters
    I_val = I_t[0]
    q_t = I_t[1] / (I_val + 1e-10)
    u_t = I_t[2] / (I_val + 1e-10)

    # Correction factor (Eq. 11.4)
    pc = 1 + m12 * (c2a * q_t + s2a * u_t) + m13 * (-s2a * q_t + c2a * u_t)

    return pc


def apply_polarization_correction_simple(
    I_measured: Union[float, np.ndarray],
    Q_rayleigh: Union[float, np.ndarray],
    U_rayleigh: Union[float, np.ndarray],
    m12: float,
    m13: float,
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Apply polarization correction to measured radiance.

    Simplified interface for applying polarization correction.

    Parameters
    ----------
    I_measured : float or ndarray
        Measured radiance.
    Q_rayleigh : float or ndarray
        Rayleigh Q component.
    U_rayleigh : float or ndarray
        Rayleigh U component.
    m12 : float
        Reduced Mueller matrix element M12/M11.
    m13 : float
        Reduced Mueller matrix element M13/M11.
    alpha : float or ndarray
        Rotation angle [degrees].

    Returns
    -------
    float or ndarray
        Corrected radiance.
    """
    alpha_rad = np.deg2rad(alpha)
    c2a = np.cos(2 * alpha_rad)
    s2a = np.sin(2 * alpha_rad)

    # Reduced Stokes parameters
    q_t = Q_rayleigh / (I_measured + 1e-10)
    u_t = U_rayleigh / (I_measured + 1e-10)

    # Correction factor
    pc = 1 + m12 * (c2a * q_t + s2a * u_t) + m13 * (-s2a * q_t + c2a * u_t)

    # Corrected radiance
    I_corrected = I_measured / pc

    return I_corrected


def meridional_angle(
    solar_zenith: Union[float, np.ndarray],
    solar_azimuth: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    view_azimuth: Union[float, np.ndarray],
    sensor_orientation: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute angle α between meridional plane and sensor reference.

    The meridional plane is defined by the local vertical (surface normal)
    and the direction of propagation of the radiance.

    Parameters
    ----------
    solar_zenith : float or ndarray
        Solar zenith angle [degrees].
    solar_azimuth : float or ndarray
        Solar azimuth angle [degrees].
    view_zenith : float or ndarray
        Sensor view zenith angle [degrees].
    view_azimuth : float or ndarray
        Sensor view azimuth angle [degrees].
    sensor_orientation : float or ndarray, optional
        Sensor roll angle relative to orbital plane [degrees].
        Default is 0.

    Returns
    -------
    float or ndarray
        Angle α between meridional plane and sensor reference [degrees].
    """
    # Convert to radians
    theta_v = np.deg2rad(view_zenith)
    phi_v = np.deg2rad(view_azimuth)

    # Unit vector in viewing direction
    # Pointing from surface to sensor
    ix = np.sin(theta_v) * np.cos(phi_v)
    iy = np.sin(theta_v) * np.sin(phi_v)
    iz = np.cos(theta_v)

    # Local vertical (surface normal) is [0, 0, 1]
    # Meridional plane contains [0, 0, 1] and [ix, iy, iz]

    # Reference direction parallel to meridional plane (l_t = -theta_hat)
    # and perpendicular (r_t = phi_hat)

    # For a sensor with axes aligned with orbital plane,
    # the angle α depends on the satellite orbital geometry

    # Simplified calculation assuming nadir-pointing sensor
    # The angle is approximately the azimuthal angle relative to orbit
    alpha = view_azimuth + sensor_orientation

    return alpha


@dataclass
class SensorPolarization:
    """
    Sensor polarization sensitivity parameters.

    The reduced Mueller matrix elements m12 and m13 describe the
    sensor's sensitivity to linear polarization.

    Attributes
    ----------
    m12 : ndarray
        Reduced Mueller matrix element M12/M11 for each band.
    m13 : ndarray
        Reduced Mueller matrix element M13/M11 for each band.
    wavelengths : ndarray
        Center wavelengths of each band [nm].
    sensor : str
        Sensor name.

    Notes
    -----
    For a sensor with no polarization sensitivity, m12 = m13 = 0.
    Typical values for MODIS are |m12| < 0.054.
    """

    m12: np.ndarray
    m13: np.ndarray
    wavelengths: np.ndarray
    sensor: str

    @classmethod
    def from_sensor(cls, sensor: str) -> "SensorPolarization":
        """
        Load polarization parameters for a sensor.

        Parameters
        ----------
        sensor : str
            Sensor name ('seawifs', 'modis_aqua', 'modis_terra',
            'viirs_npp', 'viirs_noaa20').

        Returns
        -------
        SensorPolarization
            Polarization parameters for the sensor.
        """
        sensor = sensor.lower()

        if sensor == "seawifs":
            # SeaWiFS has negligible polarization sensitivity (< 0.25%)
            wavelengths = np.array([412, 443, 490, 510, 555, 670, 765, 865])
            m12 = np.zeros(8)
            m13 = np.zeros(8)

        elif sensor in ("modis_aqua", "modis_terra"):
            # MODIS has significant polarization sensitivity
            # Values vary across detector and mirror side
            # These are approximate band-averaged values
            wavelengths = np.array([412, 443, 469, 488, 531, 547, 555, 645,
                                   667, 678, 748, 859, 869])
            # Approximate values from Meister et al. 2005
            m12 = np.array([0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                           0.02, 0.02, 0.03, 0.04, 0.04])
            m13 = np.array([0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.01,
                           0.01, 0.01, 0.01, 0.02, 0.02])

        elif sensor in ("viirs_npp", "viirs_noaa20"):
            # VIIRS has lower polarization sensitivity than MODIS
            wavelengths = np.array([412, 443, 486, 551, 671, 745, 862])
            m12 = np.array([0.015, 0.015, 0.01, 0.01, 0.01, 0.015, 0.02])
            m13 = np.array([0.005, 0.005, 0.003, 0.003, 0.003, 0.005, 0.01])

        else:
            raise ValueError(f"Unknown sensor: {sensor}")

        return cls(
            m12=m12,
            m13=m13,
            wavelengths=wavelengths,
            sensor=sensor,
        )


def compute_rayleigh_stokes(
    wavelength: float,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    relative_azimuth: Union[float, np.ndarray],
    pressure: float = 1013.25,
    wind_speed: float = 5.0,
) -> StokesVector:
    """
    Compute Rayleigh-scattered Stokes vector at TOA.

    The Rayleigh contribution dominates the TOA polarization for
    most ocean color observation conditions.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    solar_zenith : float or ndarray
        Solar zenith angle [degrees].
    view_zenith : float or ndarray
        View zenith angle [degrees].
    relative_azimuth : float or ndarray
        Relative azimuth angle [degrees].
    pressure : float, optional
        Sea level pressure [hPa]. Default is 1013.25.
    wind_speed : float, optional
        Wind speed [m/s]. Default is 5.0.

    Returns
    -------
    StokesVector
        TOA Rayleigh Stokes vector in reflectance units.

    Notes
    -----
    In operational processing, these values come from precomputed
    lookup tables generated by vector radiative transfer codes.
    This function provides an approximate analytical calculation.
    """
    # Convert angles to radians
    theta_s = np.deg2rad(solar_zenith)
    theta_v = np.deg2rad(view_zenith)
    phi = np.deg2rad(relative_azimuth)

    # Scattering angle
    cos_scatter = -np.cos(theta_s) * np.cos(theta_v) + \
                  np.sin(theta_s) * np.sin(theta_v) * np.cos(phi)
    scatter_angle = np.arccos(np.clip(cos_scatter, -1, 1))

    # Rayleigh phase function (unpolarized component)
    # P(Θ) = 3/4 * (1 + cos²Θ)
    P_unpol = 0.75 * (1 + cos_scatter**2)

    # Rayleigh optical thickness
    from . import rayleigh
    tau_r = rayleigh.rayleigh_optical_thickness(wavelength, pressure)

    # Air mass factor
    M = 1 / np.cos(theta_s) + 1 / np.cos(theta_v)

    # Single-scattering approximation for I
    I_r = tau_r * P_unpol / (4 * np.cos(theta_v))

    # Degree of polarization for Rayleigh scattering
    # P_pol(Θ) = (1 - cos²Θ) / (1 + cos²Θ)
    sin2_scatter = 1 - cos_scatter**2
    dolp = sin2_scatter / (1 + cos_scatter**2 + 1e-10)

    # Q and U components depend on scattering geometry
    # Simplified calculation
    Q_r = -I_r * dolp * np.cos(2 * phi)
    U_r = -I_r * dolp * np.sin(2 * phi)

    return StokesVector(I=I_r, Q=Q_r, U=U_r, V=0.0)


def compute_glint_stokes(
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    relative_azimuth: Union[float, np.ndarray],
    wind_speed: float = 5.0,
) -> StokesVector:
    """
    Compute sun glint Stokes vector.

    Sea surface glint can be highly polarized due to Fresnel reflection.

    Parameters
    ----------
    solar_zenith : float or ndarray
        Solar zenith angle [degrees].
    view_zenith : float or ndarray
        View zenith angle [degrees].
    relative_azimuth : float or ndarray
        Relative azimuth angle [degrees].
    wind_speed : float, optional
        Wind speed [m/s]. Default is 5.0.

    Returns
    -------
    StokesVector
        Sun glint Stokes vector.

    Notes
    -----
    Fresnel reflection produces 100% polarization at Brewster's angle.
    For water (n ≈ 1.34), Brewster's angle is about 53°.
    """
    from . import glint

    # Normalized glint radiance
    L_GN = glint.normalized_sun_glint(
        solar_zenith, view_zenith, relative_azimuth, wind_speed
    )

    # Incident angle for specular reflection
    # For specular geometry, this is related to the facet slope
    theta_s = np.deg2rad(solar_zenith)
    theta_v = np.deg2rad(view_zenith)
    phi = np.deg2rad(relative_azimuth)

    # Specular reflection angle
    cos_scatter = np.cos(theta_s) * np.cos(theta_v) + \
                  np.sin(theta_s) * np.sin(theta_v) * np.cos(phi)
    half_angle = np.arccos(np.clip(np.abs(cos_scatter), 0, 1)) / 2

    # Fresnel reflectances for s and p polarizations
    n_water = 1.34
    cos_i = np.cos(half_angle)
    sin_i = np.sin(half_angle)
    cos_t = np.sqrt(1 - (sin_i / n_water)**2)

    rs = ((cos_i - n_water * cos_t) / (cos_i + n_water * cos_t + 1e-10))**2
    rp = ((n_water * cos_i - cos_t) / (n_water * cos_i + cos_t + 1e-10))**2

    I_g = L_GN * (rs + rp) / 2
    Q_g = L_GN * (rs - rp) / 2

    return StokesVector(I=I_g, Q=Q_g, U=0.0, V=0.0)


def polarization_correction(
    Im: Union[float, np.ndarray],
    sensor_pol: SensorPolarization,
    stokes_rayleigh: StokesVector,
    alpha: Union[float, np.ndarray],
    wavelength_idx: int,
    stokes_glint: Optional[StokesVector] = None,
) -> Union[float, np.ndarray]:
    """
    Compute polarization-corrected TOA radiance.

    Applies Equation 11.6 from NASA TM-2016-217551 to correct the
    measured radiance for sensor polarization sensitivity.

    Parameters
    ----------
    Im : float or ndarray
        Measured radiance from sensor.
    sensor_pol : SensorPolarization
        Sensor polarization sensitivity parameters.
    stokes_rayleigh : StokesVector
        Rayleigh-scattered Stokes vector at TOA.
    alpha : float or ndarray
        Angle between meridional plane and sensor reference [degrees].
    wavelength_idx : int
        Index of wavelength band.
    stokes_glint : StokesVector, optional
        Sun glint Stokes vector. If None, glint is ignored.

    Returns
    -------
    float or ndarray
        Polarization-corrected TOA radiance It.

    Notes
    -----
    The correction is (Eq. 11.6):

    .. math::

        I_t = I_m - m_{12}[\\cos(2\\alpha)Q_R + \\sin(2\\alpha)U_R]
                  - m_{13}[-\\sin(2\\alpha)Q_R + \\cos(2\\alpha)U_R]

    where QR and UR are the Rayleigh Stokes components (approximating
    the total TOA polarization).
    """
    # Get sensor parameters for this band
    m12 = sensor_pol.m12[wavelength_idx]
    m13 = sensor_pol.m13[wavelength_idx]

    # Convert alpha to radians
    alpha_rad = np.deg2rad(alpha)
    c2a = np.cos(2 * alpha_rad)
    s2a = np.sin(2 * alpha_rad)

    # Total Q and U (Rayleigh dominates)
    Q_tot = stokes_rayleigh.Q
    U_tot = stokes_rayleigh.U

    if stokes_glint is not None:
        Q_tot = Q_tot + stokes_glint.Q
        U_tot = U_tot + stokes_glint.U

    # Apply correction (Eq. 11.6)
    It = Im - m12 * (c2a * Q_tot + s2a * U_tot) \
            - m13 * (-s2a * Q_tot + c2a * U_tot)

    return It


def polarization_correction_factor(
    sensor_pol: SensorPolarization,
    stokes_rayleigh: StokesVector,
    alpha: Union[float, np.ndarray],
    wavelength_idx: int,
) -> Union[float, np.ndarray]:
    """
    Compute polarization correction factor pc = Im/It.

    Parameters
    ----------
    sensor_pol : SensorPolarization
        Sensor polarization sensitivity parameters.
    stokes_rayleigh : StokesVector
        Rayleigh-scattered Stokes vector at TOA.
    alpha : float or ndarray
        Angle between meridional plane and sensor reference [degrees].
    wavelength_idx : int
        Index of wavelength band.

    Returns
    -------
    float or ndarray
        Polarization correction factor. Multiply measured radiance
        by this factor to get corrected radiance.

    Notes
    -----
    The correction factor is typically in the range 0.97 to 1.03
    for MODIS-type sensors.
    """
    m12 = sensor_pol.m12[wavelength_idx]
    m13 = sensor_pol.m13[wavelength_idx]

    alpha_rad = np.deg2rad(alpha)
    c2a = np.cos(2 * alpha_rad)
    s2a = np.sin(2 * alpha_rad)

    q_r = stokes_rayleigh.q
    u_r = stokes_rayleigh.u

    # Correction factor (inverse of Eq. 11.4 denominator)
    denominator = 1 + m12 * (c2a * q_r + s2a * u_r) \
                    + m13 * (-s2a * q_r + c2a * u_r)

    pc = 1.0 / denominator

    return pc


def apply_polarization_correction(
    Lt: np.ndarray,
    wavelengths: np.ndarray,
    solar_zenith: Union[float, np.ndarray],
    view_zenith: Union[float, np.ndarray],
    relative_azimuth: Union[float, np.ndarray],
    sensor: str,
    pressure: float = 1013.25,
    wind_speed: float = 5.0,
) -> np.ndarray:
    """
    Apply full polarization correction to TOA radiances.

    This is the main entry point for polarization correction.

    Parameters
    ----------
    Lt : ndarray
        Measured TOA radiances, shape (n_bands,) or (n_bands, n_pixels).
    wavelengths : ndarray
        Center wavelengths of each band [nm].
    solar_zenith : float or ndarray
        Solar zenith angle [degrees].
    view_zenith : float or ndarray
        View zenith angle [degrees].
    relative_azimuth : float or ndarray
        Relative azimuth angle [degrees].
    sensor : str
        Sensor name.
    pressure : float, optional
        Sea level pressure [hPa]. Default is 1013.25.
    wind_speed : float, optional
        Wind speed [m/s]. Default is 5.0.

    Returns
    -------
    ndarray
        Polarization-corrected TOA radiances.

    Examples
    --------
    >>> Lt_corrected = apply_polarization_correction(
    ...     Lt, wavelengths, 30.0, 15.0, 90.0, 'modis_aqua'
    ... )
    """
    # Load sensor parameters
    sensor_pol = SensorPolarization.from_sensor(sensor)

    # Compute meridional angle
    # Simplified: assume sensor aligned with meridional plane
    alpha = meridional_angle(
        solar_zenith, 0.0, view_zenith, relative_azimuth
    )

    # Output array
    Lt_corrected = np.zeros_like(Lt)

    for i, wl in enumerate(wavelengths):
        # Find closest band in sensor parameters
        idx = np.argmin(np.abs(sensor_pol.wavelengths - wl))

        # Compute Rayleigh Stokes vector
        stokes_r = compute_rayleigh_stokes(
            wl, solar_zenith, view_zenith, relative_azimuth,
            pressure, wind_speed
        )

        # Compute glint Stokes vector
        stokes_g = compute_glint_stokes(
            solar_zenith, view_zenith, relative_azimuth, wind_speed
        )

        # Apply correction
        if Lt.ndim == 1:
            Lt_corrected[i] = polarization_correction(
                Lt[i], sensor_pol, stokes_r, alpha, idx, stokes_g
            )
        else:
            Lt_corrected[i, :] = polarization_correction(
                Lt[i, :], sensor_pol, stokes_r, alpha, idx, stokes_g
            )

    return Lt_corrected
