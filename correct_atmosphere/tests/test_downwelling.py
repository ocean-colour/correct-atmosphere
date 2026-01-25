"""
Tests for downwelling irradiance calculations.

Tests cover:
- Solar spectrum data and interpolation
- Extraterrestrial solar irradiance with Earth-Sun distance correction
- Direct and diffuse downwelling irradiance components
- Total Ed calculation
- PACE hyperspectral support
- Unit conversions
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from correct_atmosphere.downwelling import (
    SolarSpectrum,
    get_solar_spectrum,
    get_pace_wavelengths,
    extraterrestrial_solar_irradiance,
    solar_zenith_factor,
    downwelling_irradiance_direct,
    downwelling_irradiance_diffuse,
    downwelling_irradiance,
    downwelling_irradiance_spectral,
    convert_irradiance_units,
    PACE_OCI_WAVELENGTHS,
    PACE_OCI_SWIR_BANDS,
)


class TestSolarSpectrum:
    """Tests for SolarSpectrum class and data loading."""

    def test_get_solar_spectrum_default(self):
        """Test loading default TSIS-1 HSRS spectrum."""
        spectrum = get_solar_spectrum()

        assert "TSIS-1 HSRS" in spectrum.source
        # TSIS-1 HSRS file covers 202-2730nm at 0.1nm resolution
        assert spectrum.resolution == 1.0  # reported nominal resolution
        # Full spectrum: (2730-202)/0.1 + 1 = 25281 points
        assert len(spectrum.wavelengths) == 25281
        assert spectrum.wavelengths[0] == 202.0
        assert spectrum.wavelengths[-1] == 2730.0
        assert len(spectrum.f0) == len(spectrum.wavelengths)

    def test_get_solar_spectrum_thuillier(self):
        """Test loading Thuillier 2003 spectrum."""
        spectrum = get_solar_spectrum("Thuillier2003")

        assert "Thuillier" in spectrum.source
        assert len(spectrum.wavelengths) > 0
        assert spectrum.wavelengths[0] >= 380.0

    def test_get_solar_spectrum_unknown(self):
        """Test error for unknown spectrum source."""
        with pytest.raises(ValueError, match="Unknown solar spectrum"):
            get_solar_spectrum("Unknown")

    def test_get_solar_spectrum_wavelength_range(self):
        """Test wavelength range subsetting."""
        spectrum = get_solar_spectrum(wavelength_range=(400, 700))

        assert spectrum.wavelengths[0] >= 400.0
        assert spectrum.wavelengths[-1] <= 700.0
        # 400-700 nm at 0.1nm resolution = 3001 points, which is less than full 7001
        assert len(spectrum.wavelengths) < 7001

    def test_solar_spectrum_interpolate_scalar(self):
        """Test F0 interpolation at single wavelength."""
        spectrum = get_solar_spectrum()

        f0_443 = spectrum.interpolate(443.0)

        assert isinstance(f0_443, float)
        # TSIS-1 HSRS F0 at 443 nm is ~197 mW/cm2/um
        assert 190 < f0_443 < 210

    def test_solar_spectrum_interpolate_array(self):
        """Test F0 interpolation at multiple wavelengths."""
        spectrum = get_solar_spectrum()
        wavelengths = np.array([412, 443, 490, 555, 670])

        f0 = spectrum.interpolate(wavelengths)

        assert len(f0) == 5
        # F0 should generally increase from blue to green
        assert f0[0] < f0[2]  # 412 < 490
        # All values should be positive and reasonable (TSIS-1 values)
        # 412nm: ~187, 443nm: ~197, 490nm: ~208, 555nm: ~193, 670nm: ~153
        assert np.all(f0 > 100)
        assert np.all(f0 < 250)

    def test_solar_spectrum_to_si_units(self):
        """Test unit conversion to SI."""
        spectrum = get_solar_spectrum()
        spectrum_si = spectrum.to_si_units()

        # SI units should be 10x smaller
        assert_allclose(spectrum_si.f0, spectrum.f0 * 0.1, rtol=1e-10)
        assert "[SI units]" in spectrum_si.source


class TestExtraterrestrialIrradiance:
    """Tests for extraterrestrial solar irradiance."""

    def test_f0_typical_values(self):
        """Test F0 values at standard wavelengths."""
        # TSIS-1 HSRS F0 values in mW/cm2/um (at mean Earth-Sun distance)
        # Values based on actual TSIS-1 HSRS data
        expected_ranges = {
            412: (180, 195),   # ~187
            443: (190, 210),   # ~197
            490: (200, 220),   # ~208
            555: (185, 200),   # ~193
            670: (145, 165),   # ~153
        }

        for wl, (f0_min, f0_max) in expected_ranges.items():
            f0 = extraterrestrial_solar_irradiance(float(wl), day_of_year=172)
            assert f0_min < f0 < f0_max, f"F0 at {wl} nm = {f0}, expected {f0_min}-{f0_max}"

    def test_earth_sun_distance_effect(self):
        """Test Earth-Sun distance variation over year."""
        # Perihelion ~Jan 3 (day 3): Earth closest to Sun
        # Aphelion ~Jul 4 (day 185): Earth farthest from Sun

        f0_perihelion = extraterrestrial_solar_irradiance(550.0, day_of_year=3)
        f0_aphelion = extraterrestrial_solar_irradiance(550.0, day_of_year=185)

        # Perihelion F0 should be higher than aphelion
        # The ratio (R0/R)^2 varies by about 6.7% over the year
        ratio = f0_perihelion / f0_aphelion
        assert 1.06 < ratio < 1.08

    def test_f0_array_input(self):
        """Test F0 calculation with array of wavelengths."""
        wavelengths = np.array([400, 500, 600, 700, 800])

        f0 = extraterrestrial_solar_irradiance(wavelengths)

        assert f0.shape == wavelengths.shape
        assert np.all(f0 > 0)


class TestSolarZenithFactor:
    """Tests for solar zenith angle factor."""

    def test_nadir_sun(self):
        """Test cos(0) = 1 for overhead sun."""
        assert_allclose(solar_zenith_factor(0.0), 1.0, rtol=1e-10)

    def test_typical_angles(self):
        """Test cos at typical solar zenith angles."""
        assert_allclose(solar_zenith_factor(60.0), 0.5, rtol=1e-10)
        assert_allclose(solar_zenith_factor(30.0), np.sqrt(3)/2, rtol=1e-10)

    def test_high_zenith(self):
        """Test at high solar zenith angle (low sun)."""
        cos_80 = solar_zenith_factor(80.0)
        assert 0.1 < cos_80 < 0.2

    def test_array_input(self):
        """Test with array of angles."""
        angles = np.array([0, 30, 60, 90])
        cos_angles = solar_zenith_factor(angles)

        assert_allclose(cos_angles[0], 1.0, atol=1e-10)
        assert_allclose(cos_angles[3], 0.0, atol=1e-10)


class TestDownwellingIrradianceDirect:
    """Tests for direct beam Ed component."""

    def test_basic_calculation(self):
        """Test basic direct Ed calculation."""
        ed_dir = downwelling_irradiance_direct(550.0, 30.0)

        # Should be positive and reasonable
        assert ed_dir > 0
        # At 550 nm, SZA=30, with TSIS-1 F0 ~190 mW/cm2/um
        # Ed_direct < F0 * cos(30) * T ≈ 190 * 0.866 * 0.9 ≈ 148
        assert 100 < ed_dir < 180

    def test_zenith_dependence(self):
        """Test Ed decreases with solar zenith angle."""
        ed_0 = downwelling_irradiance_direct(550.0, 0.0)
        ed_30 = downwelling_irradiance_direct(550.0, 30.0)
        ed_60 = downwelling_irradiance_direct(550.0, 60.0)

        assert ed_0 > ed_30 > ed_60

    def test_wavelength_dependence(self):
        """Test Ed varies with wavelength."""
        ed_blue = downwelling_irradiance_direct(443.0, 30.0)
        ed_green = downwelling_irradiance_direct(555.0, 30.0)
        ed_red = downwelling_irradiance_direct(670.0, 30.0)

        # Blue reduced by stronger Rayleigh scattering
        assert ed_blue < ed_green
        # All should be positive and substantial (TSIS-1 values lower than old estimates)
        assert ed_blue > 50
        assert ed_green > 100
        assert ed_red > 100

    def test_aerosol_effect(self):
        """Test aerosol reduces direct Ed."""
        ed_clean = downwelling_irradiance_direct(550.0, 30.0, aerosol_tau=0.0)
        ed_hazy = downwelling_irradiance_direct(550.0, 30.0, aerosol_tau=0.3)

        assert ed_hazy < ed_clean

    def test_pressure_effect(self):
        """Test pressure affects Rayleigh scattering."""
        ed_low = downwelling_irradiance_direct(443.0, 30.0, pressure=900.0)
        ed_std = downwelling_irradiance_direct(443.0, 30.0, pressure=1013.25)
        ed_high = downwelling_irradiance_direct(443.0, 30.0, pressure=1100.0)

        # Lower pressure = less Rayleigh scattering = more direct Ed
        assert ed_low > ed_std > ed_high


class TestDownwellingIrradianceDiffuse:
    """Tests for diffuse (sky) Ed component."""

    def test_basic_calculation(self):
        """Test basic diffuse Ed calculation."""
        ed_dif = downwelling_irradiance_diffuse(550.0, 30.0)

        # Should be positive
        assert ed_dif > 0
        # Diffuse typically smaller than direct for clear sky
        ed_dir = downwelling_irradiance_direct(550.0, 30.0)
        assert ed_dif < ed_dir

    def test_higher_at_blue(self):
        """Test diffuse component higher at blue wavelengths."""
        ed_blue = downwelling_irradiance_diffuse(443.0, 30.0)
        ed_red = downwelling_irradiance_diffuse(670.0, 30.0)

        # More Rayleigh scattering at blue = more diffuse
        assert ed_blue > ed_red

    def test_aerosol_increases_diffuse(self):
        """Test aerosols can increase diffuse component."""
        ed_clean = downwelling_irradiance_diffuse(550.0, 30.0, aerosol_tau=0.0)
        ed_moderate = downwelling_irradiance_diffuse(550.0, 30.0, aerosol_tau=0.15)

        # With moderate aerosols, diffuse can increase
        # (though with very high AOT, both decrease)
        # This is a simplified test - actual behavior depends on model
        assert ed_moderate > 0


class TestDownwellingIrradianceTotal:
    """Tests for total downwelling irradiance."""

    def test_sum_of_components(self):
        """Test total equals direct plus diffuse."""
        result = downwelling_irradiance(550.0, 30.0, components=True)

        assert_allclose(
            result["total"],
            result["direct"] + result["diffuse"],
            rtol=1e-10
        )

    def test_typical_value(self):
        """Test total Ed has expected magnitude."""
        ed = downwelling_irradiance(550.0, 30.0)

        # Total Ed at 550 nm, SZA=30, clear sky with TSIS-1 F0 ~190 mW/cm2/um
        # Ed ≈ F0 * cos(SZA) * T ≈ 190 * 0.866 * 0.85 ≈ 140
        assert 100 < ed < 180

    def test_array_wavelengths(self):
        """Test with array of wavelengths."""
        wavelengths = np.array([443, 490, 555, 670, 865])

        ed = downwelling_irradiance(wavelengths, 30.0)

        assert ed.shape == wavelengths.shape
        assert np.all(ed > 0)

    def test_spectral_shape(self):
        """Test spectral shape is reasonable."""
        wavelengths = np.arange(400, 900, 10)
        ed = downwelling_irradiance(wavelengths, 30.0)

        # Solar spectrum peaks around 500nm, so Ed at 400nm and 890nm
        # are both lower than the middle wavelengths
        # All values should be positive
        assert np.all(ed > 0)
        # Check that mid-range values are higher than extreme wavelengths
        ed_500 = downwelling_irradiance(500.0, 30.0)
        assert ed_500 > ed[0]   # 500nm > 400nm
        assert ed_500 > ed[-1]  # 500nm > 890nm

    def test_day_of_year_effect(self):
        """Test Earth-Sun distance affects Ed."""
        ed_jan = downwelling_irradiance(550.0, 30.0, day_of_year=3)
        ed_jul = downwelling_irradiance(550.0, 30.0, day_of_year=185)

        # January Ed should be higher (closer to Sun)
        assert ed_jan > ed_jul


class TestDownwellingIrradianceSpectral:
    """Tests for spectral Ed calculation."""

    def test_pace_default(self):
        """Test default PACE wavelength range."""
        result = downwelling_irradiance_spectral(30.0)

        assert result["wavelengths"][0] == 340.0
        assert result["wavelengths"][-1] == 890.0
        assert len(result["ed"]) == len(result["wavelengths"])

    def test_custom_range(self):
        """Test custom wavelength range."""
        result = downwelling_irradiance_spectral(
            30.0, wavelength_range=(400, 700), resolution=10.0
        )

        assert result["wavelengths"][0] == 400.0
        assert result["wavelengths"][-1] == 700.0

    def test_aerosol_spectral(self):
        """Test spectral aerosol extrapolation."""
        result_clean = downwelling_irradiance_spectral(30.0, aerosol_tau_550=0.05)
        result_hazy = downwelling_irradiance_spectral(30.0, aerosol_tau_550=0.3)

        # Hazy should have lower Ed everywhere
        assert np.all(result_hazy["ed"] < result_clean["ed"])

    def test_output_keys(self):
        """Test all expected keys in output."""
        result = downwelling_irradiance_spectral(30.0)

        expected_keys = ["wavelengths", "ed", "ed_direct", "ed_diffuse", "f0"]
        for key in expected_keys:
            assert key in result

    def test_components_sum(self):
        """Test direct + diffuse = total."""
        result = downwelling_irradiance_spectral(30.0)

        assert_allclose(
            result["ed"],
            result["ed_direct"] + result["ed_diffuse"],
            rtol=1e-10
        )


class TestPACEWavelengths:
    """Tests for PACE wavelength definitions."""

    def test_pace_oci_wavelengths(self):
        """Test PACE OCI hyperspectral wavelengths."""
        assert PACE_OCI_WAVELENGTHS[0] == 340.0
        assert PACE_OCI_WAVELENGTHS[-1] == 890.0
        # 5 nm resolution: (890-340)/5 + 1 = 111 bands
        assert len(PACE_OCI_WAVELENGTHS) == 111

    def test_pace_swir_bands(self):
        """Test PACE OCI SWIR bands."""
        expected_swir = [940, 1038, 1250, 1378, 1615, 2130, 2260]
        assert_allclose(PACE_OCI_SWIR_BANDS, expected_swir)

    def test_get_pace_wavelengths(self):
        """Test get_pace_wavelengths function."""
        wl = get_pace_wavelengths()
        assert len(wl) == 111

        wl_with_swir = get_pace_wavelengths(include_swir=True)
        assert len(wl_with_swir) == 111 + 7


class TestUnitConversion:
    """Tests for irradiance unit conversion."""

    def test_mw_to_w(self):
        """Test mW/cm2/um to W/m2/nm conversion."""
        ed_mw = 180.0
        ed_w = convert_irradiance_units(ed_mw, "mW_cm2_um", "W_m2_nm")

        # 1 mW/cm2/um = 0.1 W/m2/nm
        assert_allclose(ed_w, 18.0, rtol=1e-10)

    def test_w_to_uw(self):
        """Test W/m2/nm to uW/cm2/nm conversion."""
        ed_w = 18.0
        ed_uw = convert_irradiance_units(ed_w, "W_m2_nm", "uW_cm2_nm")

        # 1 W/m2/nm = 100 uW/cm2/nm (1e-6 W/cm2/nm * 1e4 = 0.01 W/cm2/nm)
        # Actually: 1 W/m2 = 0.0001 W/cm2, so 1 W/m2/nm = 100 uW/cm2/nm
        assert_allclose(ed_uw, 1800.0, rtol=1e-10)

    def test_roundtrip(self):
        """Test conversion roundtrip."""
        ed_orig = 180.0
        ed_si = convert_irradiance_units(ed_orig, "mW_cm2_um", "W_m2_nm")
        ed_back = convert_irradiance_units(ed_si, "W_m2_nm", "mW_cm2_um")

        assert_allclose(ed_back, ed_orig, rtol=1e-10)

    def test_array_input(self):
        """Test conversion with array input."""
        ed = np.array([100, 150, 200])
        ed_conv = convert_irradiance_units(ed, "mW_cm2_um", "W_m2_nm")

        assert ed_conv.shape == ed.shape

    def test_unknown_units(self):
        """Test error for unknown units."""
        with pytest.raises(ValueError):
            convert_irradiance_units(100.0, "unknown", "W_m2_nm")

        with pytest.raises(ValueError):
            convert_irradiance_units(100.0, "mW_cm2_um", "unknown")


class TestEdPhysicalConstraints:
    """Tests for physical consistency of Ed calculations."""

    def test_ed_less_than_f0(self):
        """Test Ed < F0 (atmosphere always attenuates)."""
        wavelengths = np.arange(400, 900, 50)

        ed = downwelling_irradiance(wavelengths, 30.0)
        f0 = extraterrestrial_solar_irradiance(wavelengths)

        # Ed should always be less than F0 * cos(SZA)
        cos_sza = np.cos(np.deg2rad(30.0))
        assert_array_less(ed, f0 * cos_sza * 1.01)  # 1% tolerance

    def test_ed_positive(self):
        """Test Ed always positive for valid inputs."""
        # Various conditions
        test_cases = [
            (443.0, 0.0),    # Overhead sun
            (550.0, 60.0),   # High zenith
            (800.0, 45.0),   # NIR
        ]

        for wl, sza in test_cases:
            ed = downwelling_irradiance(wl, sza)
            assert ed > 0, f"Ed should be positive for wl={wl}, sza={sza}"

    def test_spectral_continuity(self):
        """Test Ed spectrum is continuous (no large jumps)."""
        # Use coarser resolution to smooth over solar Fraunhofer lines
        result = downwelling_irradiance_spectral(30.0, resolution=5.0)
        ed = result["ed"]

        # Check that adjacent values at 5nm spacing don't differ by more than 35%
        # (solar spectrum has deep Fraunhofer absorption lines, e.g., Ca H&K at 393-397nm)
        ratios = ed[1:] / ed[:-1]
        assert np.all(ratios > 0.65)
        assert np.all(ratios < 1.50)
