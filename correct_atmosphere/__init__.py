"""
oceanatmos: Atmospheric Correction for Satellite Ocean Color Radiometry
========================================================================

A Python implementation of the NASA Ocean Biology Processing Group (OBPG)
atmospheric correction algorithms for ocean color remote sensing.

This package implements the algorithms documented in:

    Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
    Atmospheric Correction for Satellite Ocean Color Radiometry.
    NASA/TM-2016-217551.

Main Classes
------------
AtmosphericCorrection
    Main class for performing atmospheric correction on TOA radiances.

Modules
-------
rayleigh
    Rayleigh scattering corrections for atmospheric gas molecules.
gases
    Absorption corrections for O3 and NO2.
glint
    Sun glint correction algorithms.
whitecaps
    Whitecap and foam reflectance calculations.
aerosols
    Aerosol path radiance estimation (black-pixel and non-black-pixel).
transmittance
    Direct and diffuse atmospheric transmittance.
normalization
    Normalized reflectances and BRDF corrections.
polarization
    Sensor polarization sensitivity corrections.
outofband
    Spectral out-of-band response corrections.

Example
-------
>>> from oceanatmos import AtmosphericCorrection
>>> from oceanatmos.rayleigh import rayleigh_optical_thickness
>>> tau_r = rayleigh_optical_thickness(443.0)  # nm
>>> print(f"Rayleigh optical thickness at 443 nm: {tau_r:.4f}")
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from oceanatmos.correction import AtmosphericCorrection
from oceanatmos.constants import (
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
    MEAN_EARTH_SUN_DISTANCE,
)

__all__ = [
    "AtmosphericCorrection",
    "STANDARD_PRESSURE",
    "STANDARD_TEMPERATURE",
    "MEAN_EARTH_SUN_DISTANCE",
    "__version__",
]
