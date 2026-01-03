"""
Tests for the main correction module.

Tests the high-level AtmosphericCorrection class that orchestrates
the complete atmospheric correction process.
"""

import numpy as np
import pytest

from correct_atmosphere import AtmosphericCorrection
from correct_atmosphere.correction import (
    AtmosphericCorrection as AC,
    GeometryAngles,
    AncillaryData,
    CorrectionResult,
    CorrectionFlags,
)


class TestAtmosphericCorrectionInit:
    """Tests for AtmosphericCorrection initialization."""

    def test_sensor_initialization(self):
        """Test initialization with specific sensor."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert ac.sensor == "seawifs"

    def test_available_sensors(self):
        """Test that common sensors are available."""
        for sensor in ["seawifs", "modis_aqua", "viirs_npp"]:
            ac = AtmosphericCorrection(sensor=sensor)
            assert ac.sensor == sensor

    def test_unsupported_sensor_raises(self):
        """Test that unsupported sensor raises ValueError."""
        with pytest.raises(ValueError):
            AtmosphericCorrection(sensor="unknown_sensor")

    def test_initialization_options(self):
        """Test initialization with optional parameters."""
        ac = AtmosphericCorrection(
            sensor="seawifs",
            apply_polarization=False,
            apply_brdf=False,
            apply_outofband=False,
            glint_threshold=0.01,
            max_iterations=5,
            convergence_threshold=0.05,
        )
        assert ac.apply_polarization is False
        assert ac.apply_brdf is False
        assert ac.apply_outofband is False
        assert ac.glint_threshold == 0.01
        assert ac.max_iterations == 5
        assert ac.convergence_threshold == 0.05


class TestGeometryAngles:
    """Tests for GeometryAngles dataclass."""

    def test_creation(self):
        """Test geometry creation."""
        geo = GeometryAngles(
            solar_zenith=30.0,
            solar_azimuth=120.0,
            view_zenith=15.0,
            view_azimuth=180.0,
        )
        assert geo.solar_zenith == 30.0
        assert geo.view_zenith == 15.0

    def test_relative_azimuth_computed(self):
        """Test that relative azimuth is computed if not provided."""
        geo = GeometryAngles(
            solar_zenith=30.0,
            solar_azimuth=90.0,
            view_zenith=15.0,
            view_azimuth=180.0,
        )
        assert geo.relative_azimuth == 90.0

    def test_relative_azimuth_provided(self):
        """Test explicit relative azimuth."""
        geo = GeometryAngles(
            solar_zenith=30.0,
            solar_azimuth=90.0,
            view_zenith=15.0,
            view_azimuth=180.0,
            relative_azimuth=45.0,
        )
        assert geo.relative_azimuth == 45.0

    def test_air_mass_factor(self):
        """Test air mass factor calculation."""
        geo = GeometryAngles(
            solar_zenith=0.0,
            solar_azimuth=0.0,
            view_zenith=0.0,
            view_azimuth=0.0,
        )
        # M = 1/cos(0) + 1/cos(0) = 2
        assert abs(geo.air_mass_factor - 2.0) < 1e-10


class TestAncillaryData:
    """Tests for AncillaryData dataclass."""

    def test_default_values(self):
        """Test default ancillary values."""
        anc = AncillaryData()
        assert anc.pressure == 1013.25
        assert anc.wind_speed == 5.0
        assert anc.ozone == 300.0

    def test_custom_values(self):
        """Test custom ancillary values."""
        anc = AncillaryData(
            pressure=1000.0,
            wind_speed=10.0,
            ozone=350.0,
        )
        assert anc.pressure == 1000.0
        assert anc.wind_speed == 10.0
        assert anc.ozone == 350.0


class TestCorrectionFlags:
    """Tests for CorrectionFlags dataclass."""

    def test_default_flags(self):
        """Test default flag values."""
        flags = CorrectionFlags()
        assert flags.glint_masked is False
        assert flags.atmospheric_correction_warning is False
        assert flags.negative_water_leaving is False


class TestAtmosphericCorrectionProcess:
    """Tests for the atmospheric correction process."""

    def test_wavelengths_set(self):
        """Test that wavelengths are properly set for sensor."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert len(ac.wavelengths) == 8
        assert 412 in ac.wavelengths
        assert 865 in ac.wavelengths

    def test_nir_bands_set(self):
        """Test that NIR bands are properly identified."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert ac.nir_band_short == 765
        assert ac.nir_band_long == 865

    def test_lazy_lut_loading(self):
        """Test that LUTs are lazy loaded."""
        ac = AtmosphericCorrection(sensor="seawifs")
        # LUTs should not be loaded until accessed
        assert ac._rayleigh_lut is None
        assert ac._aerosol_lut is None
        assert ac._brdf_lut is None


class TestSensorSetup:
    """Tests for sensor-specific setup."""

    def test_seawifs_bands(self):
        """Test SeaWiFS band configuration."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert len(ac.wavelengths) == 8
        assert 412 in ac.wavelengths
        assert 443 in ac.wavelengths
        assert 490 in ac.wavelengths
        assert 555 in ac.wavelengths

    def test_modis_bands(self):
        """Test MODIS band configuration."""
        ac = AtmosphericCorrection(sensor="modis_aqua")
        assert len(ac.wavelengths) == 10
        assert 412 in ac.wavelengths
        assert 869 in ac.wavelengths

    def test_viirs_bands(self):
        """Test VIIRS band configuration."""
        ac = AtmosphericCorrection(sensor="viirs_npp")
        assert len(ac.wavelengths) == 7
        assert 412 in ac.wavelengths
        assert 865 in ac.wavelengths


class TestSupportedSensors:
    """Tests for supported sensor list."""

    def test_seawifs_supported(self):
        """Test SeaWiFS is supported."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert ac is not None

    def test_modis_aqua_supported(self):
        """Test MODIS-Aqua is supported."""
        ac = AtmosphericCorrection(sensor="modis_aqua")
        assert ac is not None

    def test_modis_terra_supported(self):
        """Test MODIS-Terra is supported."""
        ac = AtmosphericCorrection(sensor="modis_terra")
        assert ac is not None

    def test_viirs_npp_supported(self):
        """Test VIIRS-NPP is supported."""
        ac = AtmosphericCorrection(sensor="viirs_npp")
        assert ac is not None

    def test_viirs_noaa20_supported(self):
        """Test VIIRS-NOAA20 is supported."""
        ac = AtmosphericCorrection(sensor="viirs_noaa20")
        assert ac is not None

    def test_case_insensitive(self):
        """Test that sensor names are case insensitive."""
        ac1 = AtmosphericCorrection(sensor="SeaWiFS")
        ac2 = AtmosphericCorrection(sensor="SEAWIFS")
        ac3 = AtmosphericCorrection(sensor="seawifs")
        assert ac1.sensor == ac2.sensor == ac3.sensor == "seawifs"
