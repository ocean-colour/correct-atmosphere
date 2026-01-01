"""
Pytest configuration and shared fixtures for oceanatmos tests.
"""

import numpy as np
import pytest


@pytest.fixture
def seawifs_bands():
    """SeaWiFS band center wavelengths."""
    return [412, 443, 490, 510, 555, 670, 765, 865]


@pytest.fixture
def modis_bands():
    """MODIS-Aqua band center wavelengths."""
    return [412, 443, 488, 531, 547, 667, 678, 748, 869]


@pytest.fixture
def viirs_bands():
    """VIIRS band center wavelengths."""
    return [411, 443, 486, 551, 671, 745, 862]


@pytest.fixture
def typical_geometry():
    """Typical sun and viewing geometry."""
    return {
        'theta_s': 30.0,  # Solar zenith angle
        'theta_v': 30.0,  # Viewing zenith angle
        'phi': 90.0,      # Relative azimuth angle
    }


@pytest.fixture
def clear_atmosphere():
    """Clear atmosphere ancillary data."""
    return {
        'pressure': 1013.25,   # hPa
        'ozone_du': 350.0,     # Dobson units
        'no2_conc': 1.0e16,    # molecules/cm²
        'wind_speed': 5.0,     # m/s
        'relative_humidity': 80.0,  # %
    }


@pytest.fixture
def typical_toa_reflectance():
    """Typical TOA reflectance spectrum for clear ocean."""
    return {
        412: 0.12,
        443: 0.10,
        490: 0.08,
        510: 0.07,
        555: 0.05,
        670: 0.03,
        765: 0.02,
        865: 0.015,
    }


@pytest.fixture
def case1_rrs():
    """Typical Case 1 water Rrs spectrum."""
    return {
        412: 0.006,
        443: 0.005,
        490: 0.004,
        510: 0.0035,
        555: 0.002,
        670: 0.0005,
        765: 0.0001,
        865: 0.00005,
    }


@pytest.fixture
def turbid_rrs():
    """Typical turbid (Case 2) water Rrs spectrum."""
    return {
        412: 0.003,
        443: 0.004,
        490: 0.006,
        510: 0.008,
        555: 0.010,
        670: 0.008,
        765: 0.002,
        865: 0.001,
    }


@pytest.fixture
def solar_irradiance():
    """Extraterrestrial solar irradiance at mean Earth-Sun distance."""
    # Approximate values in W/m²/nm
    return {
        412: 172.0,
        443: 187.0,
        490: 196.0,
        510: 190.0,
        555: 186.0,
        670: 152.0,
        765: 124.0,
        865: 96.0,
    }


@pytest.fixture(params=['seawifs', 'modis_aqua', 'viirs'])
def sensor(request):
    """Parametrized fixture for different sensors."""
    return request.param


# Tolerance values for numerical comparisons
REFLECTANCE_RTOL = 0.01  # 1% relative tolerance for reflectances
RADIANCE_RTOL = 0.01     # 1% relative tolerance for radiances
ANGLE_ATOL = 0.1         # 0.1 degree absolute tolerance for angles
