"""
Tests for the constants module.

Verifies physical constants and sensor-specific parameters used
throughout the atmospheric correction process.
"""

import numpy as np
import pytest

from oceanatmos.constants import (
    # Physical constants
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
    MEAN_EARTH_SUN_DISTANCE,
    CO2_CONCENTRATION,
    DOBSON_UNIT,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    
    # Water properties
    WATER_REFRACTIVE_INDEX,
    PURE_WATER_ABSORPTION,
    PURE_WATER_BACKSCATTER,
    
    # Sensor band definitions
    SEAWIFS_BANDS,
    MODIS_BANDS,
    VIIRS_BANDS,
    
    # Ozone and NO2 cross sections
    O3_CROSS_SECTIONS,
    NO2_CROSS_SECTIONS,
    
    # Whitecap parameters
    WHITECAP_REFLECTANCE,
    WHITECAP_SPECTRAL_FACTOR,
    
    # Utility functions
    get_sensor_bands,
    get_band_averaged_value,
)


class TestPhysicalConstants:
    """Tests for physical constants."""

    def test_standard_pressure(self):
        """Standard sea level pressure is 1013.25 hPa."""
        assert STANDARD_PRESSURE == 1013.25

    def test_standard_temperature(self):
        """Standard temperature is 288.15 K."""
        assert STANDARD_TEMPERATURE == 288.15

    def test_earth_sun_distance(self):
        """Mean Earth-Sun distance is ~1 AU (149.6 million km)."""
        # In km
        assert 1.49e8 < MEAN_EARTH_SUN_DISTANCE < 1.50e8

    def test_co2_concentration(self):
        """CO2 concentration used in Rayleigh calculations."""
        # Should be around 360-420 ppm (paper uses 360)
        assert 350 < CO2_CONCENTRATION < 450

    def test_dobson_unit(self):
        """1 DU = 2.69e16 molecules/cm^2."""
        assert np.isclose(DOBSON_UNIT, 2.69e16, rtol=0.01)

    def test_speed_of_light(self):
        """Speed of light in vacuum."""
        assert np.isclose(SPEED_OF_LIGHT, 2.998e8, rtol=0.001)

    def test_planck_constant(self):
        """Planck's constant."""
        assert np.isclose(PLANCK_CONSTANT, 6.626e-34, rtol=0.001)


class TestWaterProperties:
    """Tests for pure water optical properties."""

    def test_refractive_index(self):
        """Water refractive index is approximately 1.34."""
        assert 1.33 < WATER_REFRACTIVE_INDEX < 1.35

    def test_pure_water_absorption_wavelengths(self):
        """Pure water absorption defined at standard wavelengths."""
        # Should have values at common ocean color wavelengths
        required_wavelengths = [412, 443, 490, 510, 555, 670, 765, 865]
        for wl in required_wavelengths:
            assert wl in PURE_WATER_ABSORPTION

    def test_pure_water_absorption_values(self):
        """Pure water absorption increases in red/NIR."""
        # Blue absorption should be very low
        assert PURE_WATER_ABSORPTION[443] < 0.01  # m^-1
        
        # Red absorption should be higher
        assert PURE_WATER_ABSORPTION[670] > 0.4  # m^-1
        
        # NIR absorption should be very high
        assert PURE_WATER_ABSORPTION[865] > 4.0  # m^-1

    def test_pure_water_backscatter(self):
        """Pure water backscatter decreases with wavelength."""
        # Blue backscatter
        bb_443 = PURE_WATER_BACKSCATTER[443]
        # NIR backscatter
        bb_865 = PURE_WATER_BACKSCATTER[865]
        
        # Should decrease (inverse wavelength dependence)
        assert bb_443 > bb_865

    def test_absorption_physically_reasonable(self):
        """All absorption values should be positive."""
        for wl, a_w in PURE_WATER_ABSORPTION.items():
            assert a_w >= 0, f"Negative absorption at {wl} nm"


class TestSensorBands:
    """Tests for sensor band definitions."""

    def test_seawifs_bands(self):
        """SeaWiFS has 8 bands."""
        assert len(SEAWIFS_BANDS) == 8
        
        # Check nominal wavelengths
        nominal = [412, 443, 490, 510, 555, 670, 765, 865]
        for band in SEAWIFS_BANDS:
            assert band['center'] in nominal or abs(band['center'] - nominal[0]) < 5

    def test_modis_bands(self):
        """MODIS Aqua ocean color bands."""
        assert len(MODIS_BANDS) >= 8
        
        # MODIS has specific band numbers
        band_numbers = [band.get('number') for band in MODIS_BANDS if 'number' in band]
        # Should include bands 8-16 for ocean color
        assert any(8 <= n <= 16 for n in band_numbers if n is not None)

    def test_viirs_bands(self):
        """VIIRS ocean color bands."""
        assert len(VIIRS_BANDS) >= 5
        
        # VIIRS uses M-band designations
        band_names = [band.get('name', '') for band in VIIRS_BANDS]
        assert any('M' in name for name in band_names)

    def test_band_structure(self):
        """All bands have required fields."""
        required_fields = ['center', 'fwhm']
        
        for bands in [SEAWIFS_BANDS, MODIS_BANDS, VIIRS_BANDS]:
            for band in bands:
                for field in required_fields:
                    assert field in band, f"Missing {field} in band definition"

    def test_get_sensor_bands(self):
        """get_sensor_bands returns correct bands for sensor name."""
        sw_bands = get_sensor_bands('seawifs')
        assert sw_bands == SEAWIFS_BANDS
        
        modis_bands = get_sensor_bands('modis_aqua')
        assert modis_bands == MODIS_BANDS
        
        viirs_bands = get_sensor_bands('viirs')
        assert viirs_bands == VIIRS_BANDS

    def test_get_sensor_bands_case_insensitive(self):
        """Sensor name lookup is case-insensitive."""
        assert get_sensor_bands('SeaWiFS') == get_sensor_bands('seawifs')
        assert get_sensor_bands('MODIS_Aqua') == get_sensor_bands('modis_aqua')


class TestGasCrossSections:
    """Tests for gas absorption cross sections."""

    def test_o3_cross_sections(self):
        """Ozone cross sections at standard wavelengths."""
        # O3 has Chappuis band in visible
        assert 443 in O3_CROSS_SECTIONS or 440 in O3_CROSS_SECTIONS
        
        # Cross section should be positive
        for wl, sigma in O3_CROSS_SECTIONS.items():
            assert sigma >= 0

    def test_no2_cross_sections(self):
        """NO2 cross sections at standard wavelengths."""
        # NO2 absorbs mainly in blue
        assert 412 in NO2_CROSS_SECTIONS or 410 in NO2_CROSS_SECTIONS
        
        # Blue absorption should be stronger than red
        blue_sigma = NO2_CROSS_SECTIONS.get(412, NO2_CROSS_SECTIONS.get(410, 0))
        red_sigma = NO2_CROSS_SECTIONS.get(670, NO2_CROSS_SECTIONS.get(680, 0))
        
        if blue_sigma > 0 and red_sigma > 0:
            assert blue_sigma > red_sigma

    def test_get_band_averaged_value(self):
        """Band-averaged values interpolate correctly."""
        # Test with O3 cross sections
        if 443 in O3_CROSS_SECTIONS and 490 in O3_CROSS_SECTIONS:
            sigma_443 = O3_CROSS_SECTIONS[443]
            sigma_490 = O3_CROSS_SECTIONS[490]
            
            # Interpolated value should be between
            sigma_465 = get_band_averaged_value(O3_CROSS_SECTIONS, 465.0)
            
            min_val = min(sigma_443, sigma_490)
            max_val = max(sigma_443, sigma_490)
            
            assert min_val <= sigma_465 <= max_val


class TestWhitecapParameters:
    """Tests for whitecap reflectance parameters."""

    def test_whitecap_reflectance(self):
        """Whitecap effective reflectance is 0.22."""
        assert np.isclose(WHITECAP_REFLECTANCE, 0.22, rtol=0.01)

    def test_spectral_factor_visible(self):
        """Whitecap spectral factor is 1.0 in visible."""
        visible_wavelengths = [412, 443, 490, 510, 555]
        for wl in visible_wavelengths:
            if wl in WHITECAP_SPECTRAL_FACTOR:
                assert WHITECAP_SPECTRAL_FACTOR[wl] == 1.0

    def test_spectral_factor_nir(self):
        """Whitecap spectral factor decreases in NIR."""
        # 670 nm factor
        if 670 in WHITECAP_SPECTRAL_FACTOR:
            assert WHITECAP_SPECTRAL_FACTOR[670] < 1.0
        
        # 865 nm factor should be lower still
        if 865 in WHITECAP_SPECTRAL_FACTOR:
            assert WHITECAP_SPECTRAL_FACTOR[865] < WHITECAP_SPECTRAL_FACTOR.get(670, 1.0)

    def test_frouin_values(self):
        """Values match Frouin et al. (1996)."""
        expected = {
            412: 1.0,
            443: 1.0,
            490: 1.0,
            510: 1.0,
            555: 1.0,
            670: 0.889,
            765: 0.760,
            865: 0.645,
        }
        
        for wl, expected_val in expected.items():
            if wl in WHITECAP_SPECTRAL_FACTOR:
                assert np.isclose(WHITECAP_SPECTRAL_FACTOR[wl], expected_val, rtol=0.01)


class TestConsistency:
    """Tests for consistency between related constants."""

    def test_absorption_backscatter_wavelengths_match(self):
        """Absorption and backscatter defined at same wavelengths."""
        a_wavelengths = set(PURE_WATER_ABSORPTION.keys())
        bb_wavelengths = set(PURE_WATER_BACKSCATTER.keys())
        
        # Should have significant overlap
        common = a_wavelengths & bb_wavelengths
        assert len(common) >= 5

    def test_sensor_bands_cover_visible_nir(self):
        """All sensors cover visible to NIR range."""
        for bands in [SEAWIFS_BANDS, MODIS_BANDS, VIIRS_BANDS]:
            centers = [b['center'] for b in bands]
            
            # Should have blue bands (400-450 nm)
            assert any(400 <= c <= 450 for c in centers)
            
            # Should have NIR bands (750-900 nm)
            assert any(750 <= c <= 900 for c in centers)

    def test_cross_sections_positive(self):
        """All cross sections are non-negative."""
        for sigma_dict in [O3_CROSS_SECTIONS, NO2_CROSS_SECTIONS]:
            for wl, sigma in sigma_dict.items():
                assert sigma >= 0, f"Negative cross section at {wl} nm"


class TestUnits:
    """Tests to verify correct units are used."""

    def test_pressure_units(self):
        """Pressure in hPa (hectopascals = millibars)."""
        # 1 atm = 1013.25 hPa
        assert 1000 < STANDARD_PRESSURE < 1020

    def test_temperature_units(self):
        """Temperature in Kelvin."""
        # ~15°C = 288 K
        assert 280 < STANDARD_TEMPERATURE < 300

    def test_absorption_units(self):
        """Pure water absorption in m^-1."""
        # 670 nm absorption is ~0.44 m^-1
        if 670 in PURE_WATER_ABSORPTION:
            a_670 = PURE_WATER_ABSORPTION[670]
            assert 0.3 < a_670 < 0.6

    def test_cross_section_units(self):
        """Cross sections in cm^2/molecule."""
        # O3 cross section at 443 nm is ~10^-21 cm^2/molecule
        if 443 in O3_CROSS_SECTIONS:
            sigma = O3_CROSS_SECTIONS[443]
            assert 1e-23 < sigma < 1e-19


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
