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

#: PACE OCI hyperspectral configuration
#: OCI provides continuous coverage 340-890 nm at 5 nm resolution (111 bands)
#: plus 7 discrete SWIR bands for atmospheric correction
PACE_OCI_BANDS: Dict[str, Tuple[float, float, float]] = {
    # Hyperspectral UV-VIS-NIR bands (5 nm resolution, 340-890 nm)
    # Representative entries shown; full coverage is continuous
    "340": (340.0, 337.5, 342.5),
    "345": (345.0, 342.5, 347.5),
    "350": (350.0, 347.5, 352.5),
    "355": (355.0, 352.5, 357.5),
    "360": (360.0, 357.5, 362.5),
    "365": (365.0, 362.5, 367.5),
    "370": (370.0, 367.5, 372.5),
    "375": (375.0, 372.5, 377.5),
    "380": (380.0, 377.5, 382.5),
    "385": (385.0, 382.5, 387.5),
    "390": (390.0, 387.5, 392.5),
    "395": (395.0, 392.5, 397.5),
    "400": (400.0, 397.5, 402.5),
    "405": (405.0, 402.5, 407.5),
    "410": (410.0, 407.5, 412.5),
    "415": (415.0, 412.5, 417.5),
    "420": (420.0, 417.5, 422.5),
    "425": (425.0, 422.5, 427.5),
    "430": (430.0, 427.5, 432.5),
    "435": (435.0, 432.5, 437.5),
    "440": (440.0, 437.5, 442.5),
    "443": (443.0, 440.5, 445.5),  # Heritage ocean color band
    "445": (445.0, 442.5, 447.5),
    "450": (450.0, 447.5, 452.5),
    "455": (455.0, 452.5, 457.5),
    "460": (460.0, 457.5, 462.5),
    "465": (465.0, 462.5, 467.5),
    "470": (470.0, 467.5, 472.5),
    "475": (475.0, 472.5, 477.5),
    "480": (480.0, 477.5, 482.5),
    "485": (485.0, 482.5, 487.5),
    "490": (490.0, 487.5, 492.5),
    "495": (495.0, 492.5, 497.5),
    "500": (500.0, 497.5, 502.5),
    "505": (505.0, 502.5, 507.5),
    "510": (510.0, 507.5, 512.5),
    "515": (515.0, 512.5, 517.5),
    "520": (520.0, 517.5, 522.5),
    "525": (525.0, 522.5, 527.5),
    "530": (530.0, 527.5, 532.5),
    "535": (535.0, 532.5, 537.5),
    "540": (540.0, 537.5, 542.5),
    "545": (545.0, 542.5, 547.5),
    "550": (550.0, 547.5, 552.5),
    "555": (555.0, 552.5, 557.5),
    "560": (560.0, 557.5, 562.5),
    "565": (565.0, 562.5, 567.5),
    "570": (570.0, 567.5, 572.5),
    "575": (575.0, 572.5, 577.5),
    "580": (580.0, 577.5, 582.5),
    "585": (585.0, 582.5, 587.5),
    "590": (590.0, 587.5, 592.5),
    "595": (595.0, 592.5, 597.5),
    "600": (600.0, 597.5, 602.5),
    "605": (605.0, 602.5, 607.5),
    "610": (610.0, 607.5, 612.5),
    "615": (615.0, 612.5, 617.5),
    "620": (620.0, 617.5, 622.5),
    "625": (625.0, 622.5, 627.5),
    "630": (630.0, 627.5, 632.5),
    "635": (635.0, 632.5, 637.5),
    "640": (640.0, 637.5, 642.5),
    "645": (645.0, 642.5, 647.5),
    "650": (650.0, 647.5, 652.5),
    "655": (655.0, 652.5, 657.5),
    "660": (660.0, 657.5, 662.5),
    "665": (665.0, 662.5, 667.5),
    "670": (670.0, 667.5, 672.5),
    "675": (675.0, 672.5, 677.5),
    "680": (680.0, 677.5, 682.5),
    "685": (685.0, 682.5, 687.5),
    "690": (690.0, 687.5, 692.5),
    "695": (695.0, 692.5, 697.5),
    "700": (700.0, 697.5, 702.5),
    "705": (705.0, 702.5, 707.5),
    "710": (710.0, 707.5, 712.5),
    "715": (715.0, 712.5, 717.5),
    "720": (720.0, 717.5, 722.5),
    "725": (725.0, 722.5, 727.5),
    "730": (730.0, 727.5, 732.5),
    "735": (735.0, 732.5, 737.5),
    "740": (740.0, 737.5, 742.5),
    "745": (745.0, 742.5, 747.5),
    "750": (750.0, 747.5, 752.5),
    "755": (755.0, 752.5, 757.5),
    "760": (760.0, 757.5, 762.5),  # O2-A band
    "765": (765.0, 762.5, 767.5),
    "770": (770.0, 767.5, 772.5),
    "775": (775.0, 772.5, 777.5),
    "780": (780.0, 777.5, 782.5),
    "785": (785.0, 782.5, 787.5),
    "790": (790.0, 787.5, 792.5),
    "795": (795.0, 792.5, 797.5),
    "800": (800.0, 797.5, 802.5),
    "805": (805.0, 802.5, 807.5),
    "810": (810.0, 807.5, 812.5),
    "815": (815.0, 812.5, 817.5),
    "820": (820.0, 817.5, 822.5),
    "825": (825.0, 822.5, 827.5),
    "830": (830.0, 827.5, 832.5),
    "835": (835.0, 832.5, 837.5),
    "840": (840.0, 837.5, 842.5),
    "845": (845.0, 842.5, 847.5),
    "850": (850.0, 847.5, 852.5),
    "855": (855.0, 852.5, 857.5),
    "860": (860.0, 857.5, 862.5),
    "865": (865.0, 862.5, 867.5),
    "870": (870.0, 867.5, 872.5),
    "875": (875.0, 872.5, 877.5),
    "880": (880.0, 877.5, 882.5),
    "885": (885.0, 882.5, 887.5),
    "890": (890.0, 887.5, 892.5),
    # SWIR bands for atmospheric correction
    "940": (940.0, 930.0, 950.0),    # H2O absorption
    "1038": (1038.0, 1028.0, 1048.0),
    "1250": (1250.0, 1230.0, 1270.0),
    "1378": (1378.0, 1368.0, 1388.0),  # Cirrus detection
    "1615": (1615.0, 1595.0, 1635.0),
    "2130": (2130.0, 2105.0, 2155.0),
    "2260": (2260.0, 2235.0, 2285.0),
}

#: NIR reference bands for aerosol correction (Table 9.1)
NIR_BANDS: Dict[str, Tuple[str, str]] = {
    # sensor: (lambda_1_band, lambda_2_band)
    "SeaWiFS": ("765", "865"),
    "MODIS-Aqua": ("748", "869"),
    "MODIS-Terra": ("748", "869"),
    "VIIRS-NPP": ("M6", "M7"),
    "VIIRS-NOAA20": ("M6", "M7"),
    "PACE-OCI": ("865", "1038"),  # Uses hyperspectral NIR + SWIR
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
    "PACE": {
        # PACE OCI hyperspectral: 340-890 nm at 5 nm resolution (111 bands)
        # Plus 7 SWIR bands for atmospheric correction
        "center_wavelengths": list(range(340, 895, 5)) + [940, 1038, 1250, 1378, 1615, 2130, 2260],
        "bandwidths": [5] * 111 + [20, 20, 40, 20, 40, 50, 50],
        "hyperspectral": True,
        "hyperspectral_range": (340, 890),
        "hyperspectral_resolution": 5,
    },
}

#: NIR reference bands wavelengths for aerosol correction
NIR_REFERENCE_BANDS: Dict[str, Dict[str, int]] = {
    "SEAWIFS": {"short": 765, "long": 865},
    "MODIS": {"short": 748, "long": 869},
    "VIIRS": {"short": 746, "long": 865},
    "PACE": {"short": 865, "long": 1038},
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
    "555": 0.0943,
    "645": 0.0520,
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

#: Band-averaged O3 absorption cross sections for MODIS [cm^2/molecule]
#: Values × 10^21 for convenience
O3_CROSS_SECTION_MODIS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-21
    "412": 0.000,
    "443": 0.003,
    "469": 0.009,
    "488": 0.019,
    "531": 0.065,
    "547": 0.087,
    "555": 0.106,
    "645": 0.060,
    "667": 0.050,
    "678": 0.045,
    "748": 0.012,
    "869": 0.000,
}

#: Band-averaged NO2 absorption cross sections for MODIS [cm^2/molecule]
#: Values × 10^19 for convenience
NO2_CROSS_SECTION_MODIS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-19
    "412": 0.600,
    "443": 0.560,
    "469": 0.440,
    "488": 0.340,
    "531": 0.150,
    "547": 0.100,
    "555": 0.075,
    "645": 0.012,
    "667": 0.008,
    "678": 0.006,
    "748": 0.000,
    "869": 0.000,
}

#: Band-averaged O3 absorption cross sections for VIIRS [cm^2/molecule]
#: Values × 10^21 for convenience
O3_CROSS_SECTION_VIIRS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-21
    "M1": 0.000,
    "M2": 0.004,
    "M3": 0.019,
    "M4": 0.106,
    "M5": 0.045,
    "M6": 0.012,
    "M7": 0.000,
}

#: Band-averaged NO2 absorption cross sections for VIIRS [cm^2/molecule]
#: Values × 10^19 for convenience
NO2_CROSS_SECTION_VIIRS: Dict[str, float] = {
    # Values in cm^2/molecule × 10^-19
    "M1": 0.600,
    "M2": 0.540,
    "M3": 0.340,
    "M4": 0.075,
    "M5": 0.007,
    "M6": 0.000,
    "M7": 0.000,
}

#: Unified O3 absorption cross sections by sensor
O3_CROSS_SECTION: Dict[str, Dict[str, float]] = {
    "SeaWiFS": O3_CROSS_SECTION_SEAWIFS,
    "seawifs": O3_CROSS_SECTION_SEAWIFS,
    "MODIS-Aqua": O3_CROSS_SECTION_MODIS,
    "modis_aqua": O3_CROSS_SECTION_MODIS,
    "MODIS-Terra": O3_CROSS_SECTION_MODIS,
    "modis_terra": O3_CROSS_SECTION_MODIS,
    "VIIRS-NPP": O3_CROSS_SECTION_VIIRS,
    "viirs_npp": O3_CROSS_SECTION_VIIRS,
    "VIIRS-NOAA20": O3_CROSS_SECTION_VIIRS,
    "viirs_noaa20": O3_CROSS_SECTION_VIIRS,
}

#: Unified NO2 absorption cross sections by sensor
NO2_CROSS_SECTION: Dict[str, Dict[str, float]] = {
    "SeaWiFS": NO2_CROSS_SECTION_SEAWIFS,
    "seawifs": NO2_CROSS_SECTION_SEAWIFS,
    "MODIS-Aqua": NO2_CROSS_SECTION_MODIS,
    "modis_aqua": NO2_CROSS_SECTION_MODIS,
    "MODIS-Terra": NO2_CROSS_SECTION_MODIS,
    "modis_terra": NO2_CROSS_SECTION_MODIS,
    "VIIRS-NPP": NO2_CROSS_SECTION_VIIRS,
    "viirs_npp": NO2_CROSS_SECTION_VIIRS,
    "VIIRS-NOAA20": NO2_CROSS_SECTION_VIIRS,
    "viirs_noaa20": NO2_CROSS_SECTION_VIIRS,
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
