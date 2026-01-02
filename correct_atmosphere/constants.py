"""
Physical constants and sensor parameters for atmospheric correction.

This module contains constants used throughout the atmospheric correction
algorithms, including:

- Physical constants (standard atmosphere, Earth-Sun parameters)
- Sensor-specific band information (SeaWiFS, MODIS, VIIRS)
- Band-averaged optical properties
- Pure water optical properties

References
----------
.. [1] Mobley et al. (2016), NASA/TM-2016-217551
.. [2] Bodhaine et al. (1999), J. Atmos. Oceanic Technol., 16:1854-1861
"""

import numpy as np
from typing import Dict, Tuple

# =============================================================================
# Physical Constants
# =============================================================================

#: Standard sea level atmospheric pressure [hPa]
STANDARD_PRESSURE: float = 1013.25

#: Standard temperature [K] for Rayleigh calculations
STANDARD_TEMPERATURE: float = 288.15

#: Standard CO2 concentration [ppm] for Rayleigh calculations
STANDARD_CO2_PPM: float = 360.0

#: Mean Earth-Sun distance [AU]
MEAN_EARTH_SUN_DISTANCE: float = 1.0

#: Speed of light in vacuum [m/s]
SPEED_OF_LIGHT: float = 2.998e8

#: Refractive index of pure water (approximate, visible wavelengths)
WATER_REFRACTIVE_INDEX: float = 1.34

# =============================================================================
# Sensor Band Definitions
# =============================================================================

#: SeaWiFS nominal band wavelengths [nm] and bandwidths
SEAWIFS_BANDS: Dict[str, Tuple[float, float, float]] = {
    # band_name: (center_wavelength, lower_bound, upper_bound)
    "412": (412.0, 402.0, 422.0),
    "443": (443.0, 433.0, 453.0),
    "490": (490.0, 480.0, 500.0),
    "510": (510.0, 500.0, 520.0),
    "555": (555.0, 545.0, 565.0),
    "670": (670.0, 660.0, 680.0),
    "765": (765.0, 745.0, 785.0),  # NIR band 1 (lambda_1)
    "865": (865.0, 845.0, 885.0),  # NIR band 2 (lambda_2)
}

#: MODIS-Aqua nominal band wavelengths [nm]
MODIS_AQUA_BANDS: Dict[str, Tuple[float, float, float]] = {
    "412": (412.0, 405.0, 420.0),
    "443": (443.0, 438.0, 448.0),
    "469": (469.0, 459.0, 479.0),
    "488": (488.0, 483.0, 493.0),
    "531": (531.0, 526.0, 536.0),
    "547": (547.0, 543.0, 553.0),
    "555": (555.0, 545.0, 565.0),
    "645": (645.0, 620.0, 670.0),
    "667": (667.0, 662.0, 672.0),
    "678": (678.0, 673.0, 683.0),
    "748": (748.0, 743.0, 753.0),  # NIR band 1 (lambda_1)
    "869": (869.0, 862.0, 877.0),  # NIR band 2 (lambda_2)
}

#: VIIRS-NPP nominal band wavelengths [nm]
VIIRS_BANDS: Dict[str, Tuple[float, float, float]] = {
    "M1": (412.0, 402.0, 422.0),
    "M2": (445.0, 436.0, 454.0),
    "M3": (488.0, 478.0, 498.0),
    "M4": (555.0, 545.0, 565.0),
    "M5": (672.0, 662.0, 682.0),
    "M6": (746.0, 739.0, 754.0),  # NIR band 1 (lambda_1)
    "M7": (865.0, 846.0, 885.0),  # NIR band 2 (lambda_2)
}

#: NIR reference bands for aerosol correction (Table 9.1)
NIR_BANDS: Dict[str, Tuple[str, str]] = {
    # sensor: (lambda_1_band, lambda_2_band)
    "SeaWiFS": ("765", "865"),
    "MODIS-Aqua": ("748", "869"),
    "MODIS-Terra": ("748", "869"),
    "VIIRS-NPP": ("M6", "M7"),
    "VIIRS-NOAA20": ("M6", "M7"),
}

#: Sensor band definitions for AtmosphericCorrection class
SENSOR_BANDS: Dict[str, Dict] = {
    "SEAWIFS": {
        "center_wavelengths": [412, 443, 490, 510, 555, 670, 765, 865],
        "bandwidths": [20, 20, 20, 20, 20, 20, 40, 40],
    },
    "MODIS": {
        "center_wavelengths": [412, 443, 469, 488, 531, 547, 667, 678, 748, 869],
        "bandwidths": [15, 10, 20, 10, 10, 10, 10, 10, 10, 15],
    },
    "VIIRS": {
        "center_wavelengths": [412, 445, 488, 555, 672, 746, 865],
        "bandwidths": [20, 18, 20, 20, 20, 15, 39],
    },
}

#: NIR reference bands wavelengths for aerosol correction
NIR_REFERENCE_BANDS: Dict[str, Dict[str, int]] = {
    "SEAWIFS": {"short": 765, "long": 865},
    "MODIS": {"short": 748, "long": 869},
    "VIIRS": {"short": 746, "long": 865},
}

# =============================================================================
# Band-Averaged Rayleigh Optical Thickness (Figure 6.5)
# =============================================================================

#: Band-averaged Rayleigh optical thickness at standard pressure
#: From Bodhaine et al. (1999), Eq. 30
RAYLEIGH_TAU_SEAWIFS: Dict[str, float] = {
    "412": 0.3183,
    "443": 0.2346,
    "490": 0.1559,
    "510": 0.1324,
    "555": 0.0943,
    "670": 0.0443,
    "765": 0.0271,
    "865": 0.0164,
}

RAYLEIGH_TAU_MODIS: Dict[str, float] = {
    "412": 0.3189,
    "443": 0.2357,
    "469": 0.1879,
    "488": 0.1595,
    "531": 0.1143,
    "547": 0.1009,
    "667": 0.0457,
    "678": 0.0430,
    "748": 0.0291,
    "869": 0.0159,
}

RAYLEIGH_TAU_VIIRS: Dict[str, float] = {
    "M1": 0.3164,
    "M2": 0.2289,
    "M3": 0.1567,
    "M4": 0.0944,
    "M5": 0.0438,
    "M6": 0.0283,
    "M7": 0.0163,
}

#: Unified Rayleigh optical depth by sensor
RAYLEIGH_OD: Dict[str, Dict[str, float]] = {
    "SeaWiFS": RAYLEIGH_TAU_SEAWIFS,
    "seawifs": RAYLEIGH_TAU_SEAWIFS,
    "MODIS-Aqua": RAYLEIGH_TAU_MODIS,
    "modis_aqua": RAYLEIGH_TAU_MODIS,
    "MODIS-Terra": RAYLEIGH_TAU_MODIS,
    "modis_terra": RAYLEIGH_TAU_MODIS,
    "VIIRS-NPP": RAYLEIGH_TAU_VIIRS,
    "viirs_npp": RAYLEIGH_TAU_VIIRS,
    "VIIRS-NOAA20": RAYLEIGH_TAU_VIIRS,
    "viirs_noaa20": RAYLEIGH_TAU_VIIRS,
}

# =============================================================================
# Gas Absorption Cross Sections (Figure 6.5)
# =============================================================================

#: Band-averaged O3 absorption cross sections [cm^2/molecule]
#: Values × 10^21 for convenience
O3_CROSS_SECTION_SEAWIFS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-21
    "412": 0.000,
    "443": 0.003,
    "490": 0.021,
    "510": 0.040,
    "555": 0.106,
    "670": 0.048,
    "765": 0.007,
    "865": 0.000,
}

#: Band-averaged NO2 absorption cross sections [cm^2/molecule]
#: Values × 10^19 for convenience
NO2_CROSS_SECTION_SEAWIFS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-19
    "412": 0.600,
    "443": 0.560,
    "490": 0.320,
    "510": 0.210,
    "555": 0.075,
    "670": 0.008,
    "765": 0.000,
    "865": 0.000,
}

#: Unified O3 absorption cross sections by sensor
O3_CROSS_SECTION: Dict[str, Dict[str, float]] = {
    "SeaWiFS": O3_CROSS_SECTION_SEAWIFS,
    "seawifs": O3_CROSS_SECTION_SEAWIFS,
    "MODIS-Aqua": O3_CROSS_SECTION_SEAWIFS,  # Using SeaWiFS as proxy
    "modis_aqua": O3_CROSS_SECTION_SEAWIFS,
    "MODIS-Terra": O3_CROSS_SECTION_SEAWIFS,
    "modis_terra": O3_CROSS_SECTION_SEAWIFS,
    "VIIRS-NPP": O3_CROSS_SECTION_SEAWIFS,
    "viirs_npp": O3_CROSS_SECTION_SEAWIFS,
    "VIIRS-NOAA20": O3_CROSS_SECTION_SEAWIFS,
    "viirs_noaa20": O3_CROSS_SECTION_SEAWIFS,
}

#: Unified NO2 absorption cross sections by sensor
NO2_CROSS_SECTION: Dict[str, Dict[str, float]] = {
    "SeaWiFS": NO2_CROSS_SECTION_SEAWIFS,
    "seawifs": NO2_CROSS_SECTION_SEAWIFS,
    "MODIS-Aqua": NO2_CROSS_SECTION_SEAWIFS,  # Using SeaWiFS as proxy
    "modis_aqua": NO2_CROSS_SECTION_SEAWIFS,
    "MODIS-Terra": NO2_CROSS_SECTION_SEAWIFS,
    "modis_terra": NO2_CROSS_SECTION_SEAWIFS,
    "VIIRS-NPP": NO2_CROSS_SECTION_SEAWIFS,
    "viirs_npp": NO2_CROSS_SECTION_SEAWIFS,
    "VIIRS-NOAA20": NO2_CROSS_SECTION_SEAWIFS,
    "viirs_noaa20": NO2_CROSS_SECTION_SEAWIFS,
}

# =============================================================================
# Pure Water Optical Properties
# =============================================================================

#: Pure water absorption coefficient [m^-1] at various wavelengths
#: Used in non-black-pixel algorithm (Section 9.3)
PURE_WATER_ABSORPTION: Dict[float, float] = {
    # wavelength [nm]: absorption [m^-1]
    412.0: 0.00455,
    443.0: 0.00707,
    490.0: 0.0150,
    510.0: 0.0325,
    555.0: 0.0596,
    670.0: 0.439,
    745.0: 2.806,  # VIIRS M6
    765.0: 2.85,   # SeaWiFS band 7
    862.0: 4.61,   # VIIRS M7
    865.0: 4.61,   # SeaWiFS band 8
}

#: Pure seawater backscattering coefficient [m^-1]
#: bbw(lambda) = bbw(500) * (500/lambda)^4.32 approximately
PURE_WATER_BACKSCATTER: Dict[float, float] = {
    # wavelength [nm]: backscatter [m^-1]
    412.0: 0.000877,
    443.0: 0.000682,
    490.0: 0.000477,
    510.0: 0.000416,
    555.0: 0.000313,
    670.0: 0.000168,
    745.0: 0.000119,
    765.0: 0.000109,
    862.0: 0.0000734,
    865.0: 0.0000722,
}

#: Pure water absorption by sensor band [m^-1]
_PURE_WATER_ABS_SEAWIFS: Dict[str, float] = {
    "412": 0.00455, "443": 0.00707, "490": 0.0150, "510": 0.0325,
    "555": 0.0596, "670": 0.439, "765": 2.85, "865": 4.61,
}
_PURE_WATER_ABS_MODIS: Dict[str, float] = {
    "412": 0.00455, "443": 0.00707, "469": 0.0099, "488": 0.0145,
    "531": 0.0439, "547": 0.0530, "555": 0.0596, "645": 0.325,
    "667": 0.420, "678": 0.465, "748": 2.47, "869": 4.61,
}
_PURE_WATER_ABS_VIIRS: Dict[str, float] = {
    "M1": 0.00455, "M2": 0.00707, "M3": 0.0145, "M4": 0.0596,
    "M5": 0.430, "M6": 2.806, "M7": 4.61,
}

PURE_WATER_ABS: Dict[str, Dict[str, float]] = {
    "SeaWiFS": _PURE_WATER_ABS_SEAWIFS,
    "seawifs": _PURE_WATER_ABS_SEAWIFS,
    "MODIS-Aqua": _PURE_WATER_ABS_MODIS,
    "modis_aqua": _PURE_WATER_ABS_MODIS,
    "MODIS-Terra": _PURE_WATER_ABS_MODIS,
    "modis_terra": _PURE_WATER_ABS_MODIS,
    "VIIRS-NPP": _PURE_WATER_ABS_VIIRS,
    "viirs_npp": _PURE_WATER_ABS_VIIRS,
    "VIIRS-NOAA20": _PURE_WATER_ABS_VIIRS,
    "viirs_noaa20": _PURE_WATER_ABS_VIIRS,
}

#: Pure water backscatter by sensor band [m^-1]
_PURE_WATER_BB_SEAWIFS: Dict[str, float] = {
    "412": 0.000877, "443": 0.000682, "490": 0.000477, "510": 0.000416,
    "555": 0.000313, "670": 0.000168, "765": 0.000109, "865": 0.0000722,
}
_PURE_WATER_BB_MODIS: Dict[str, float] = {
    "412": 0.000877, "443": 0.000682, "469": 0.000560, "488": 0.000485,
    "531": 0.000370, "547": 0.000330, "555": 0.000313, "645": 0.000190,
    "667": 0.000172, "678": 0.000163, "748": 0.000122, "869": 0.0000722,
}
_PURE_WATER_BB_VIIRS: Dict[str, float] = {
    "M1": 0.000877, "M2": 0.000670, "M3": 0.000485, "M4": 0.000313,
    "M5": 0.000170, "M6": 0.000119, "M7": 0.0000734,
}

PURE_WATER_BB: Dict[str, Dict[str, float]] = {
    "SeaWiFS": _PURE_WATER_BB_SEAWIFS,
    "seawifs": _PURE_WATER_BB_SEAWIFS,
    "MODIS-Aqua": _PURE_WATER_BB_MODIS,
    "modis_aqua": _PURE_WATER_BB_MODIS,
    "MODIS-Terra": _PURE_WATER_BB_MODIS,
    "modis_terra": _PURE_WATER_BB_MODIS,
    "VIIRS-NPP": _PURE_WATER_BB_VIIRS,
    "viirs_npp": _PURE_WATER_BB_VIIRS,
    "VIIRS-NOAA20": _PURE_WATER_BB_VIIRS,
    "viirs_noaa20": _PURE_WATER_BB_VIIRS,
}

# =============================================================================
# Whitecap Parameters (Chapter 8)
# =============================================================================

#: Effective whitecap irradiance reflectance (Koepke, 1984)
WHITECAP_REFLECTANCE_EFFECTIVE: float = 0.22

#: Normalized whitecap reflectance spectral dependence (Frouin et al., 1996)
#: Values are relative to blue wavelengths where a_wc = 1.0
WHITECAP_SPECTRAL_FACTOR: Dict[int, float] = {
    412: 1.000,
    443: 1.000,
    490: 1.000,
    510: 1.000,
    555: 1.000,
    670: 0.889,
    765: 0.760,
    865: 0.645,
}

#: Minimum wind speed for whitecap correction [m/s]
WHITECAP_MIN_WIND_SPEED: float = 6.33

#: Maximum wind speed for whitecap correction [m/s]
WHITECAP_MAX_WIND_SPEED: float = 12.0

# =============================================================================
# Sun Glint Parameters (Chapter 7)
# =============================================================================

#: Maximum normalized sun glint for processing [sr^-1]
#: Pixels with L_GN > this value are masked
GLINT_THRESHOLD: float = 0.005

# =============================================================================
# Aerosol Parameters (Chapter 9)
# =============================================================================

#: Relative humidity values in aerosol LUTs [%]
AEROSOL_RH_VALUES: Tuple[int, ...] = (30, 50, 70, 75, 80, 85, 90, 95)

#: Number of aerosol models in LUT
AEROSOL_NUM_MODELS: int = 10

#: Chlorophyll threshold for non-black-pixel iteration [mg/m^3]
#: No iteration if Chl < this value
NONBLACK_PIXEL_CHL_MIN: float = 0.3

#: Chlorophyll threshold for mandatory non-black-pixel iteration [mg/m^3]
#: Always iterate if Chl > this value
NONBLACK_PIXEL_CHL_MAX: float = 0.7

#: Convergence threshold for non-black-pixel iteration
NONBLACK_PIXEL_CONVERGENCE: float = 0.02  # 2%

#: Maximum iterations for non-black-pixel algorithm
NONBLACK_PIXEL_MAX_ITER: int = 10

# =============================================================================
# BRDF Correction Parameters (Section 3.2)
# =============================================================================

#: Reference value of R(theta_v, W) for normal incidence (Morel et al., 2002)
BRDF_R0: float = 0.529

#: Wavelengths for f/Q lookup tables [nm]
BRDF_WAVELENGTHS: Tuple[float, ...] = (412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 660.0)

#: Chlorophyll values for f/Q lookup tables [mg/m^3]
BRDF_CHL_VALUES: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

#: Solar zenith angles for f/Q lookup tables [degrees]
BRDF_SOLAR_ZENITH: Tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)


def get_sensor_bands(sensor: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Get band definitions for a specific sensor.

    Parameters
    ----------
    sensor : str
        Sensor name. One of 'SeaWiFS', 'MODIS-Aqua', 'MODIS-Terra',
        'VIIRS-NPP', 'VIIRS-NOAA20'.

    Returns
    -------
    dict
        Dictionary mapping band names to (center, lower, upper) wavelengths.

    Raises
    ------
    ValueError
        If sensor is not recognized.
    """
    sensor_upper = sensor.upper()
    if sensor_upper == "SEAWIFS":
        return SEAWIFS_BANDS
    elif sensor_upper in ("MODIS-AQUA", "MODIS-TERRA"):
        return MODIS_AQUA_BANDS
    elif sensor_upper in ("VIIRS-NPP", "VIIRS-NOAA20"):
        return VIIRS_BANDS
    else:
        raise ValueError(
            f"Unknown sensor: {sensor}. "
            f"Supported: SeaWiFS, MODIS-Aqua, MODIS-Terra, VIIRS-NPP, VIIRS-NOAA20"
        )


def get_nir_bands(sensor: str) -> Tuple[str, str]:
    """
    Get NIR band names for aerosol correction.

    Parameters
    ----------
    sensor : str
        Sensor name.

    Returns
    -------
    tuple of str
        (lambda_1_band, lambda_2_band) where lambda_1 < lambda_2.
    """
    sensor_key = sensor.upper()
    if sensor_key in ("MODIS-AQUA", "MODIS-TERRA"):
        sensor_key = "MODIS-Aqua" if "AQUA" in sensor_key else "MODIS-Terra"
    elif sensor_key == "SEAWIFS":
        sensor_key = "SeaWiFS"
    elif "VIIRS" in sensor_key:
        sensor_key = "VIIRS-NPP" if "NPP" in sensor_key else "VIIRS-NOAA20"
    
    if sensor_key not in NIR_BANDS:
        raise ValueError(f"Unknown sensor: {sensor}")
    
    return NIR_BANDS[sensor_key]
