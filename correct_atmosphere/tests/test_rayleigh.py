"""
Tests for the rayleigh module.

Tests Rayleigh scattering calculations based on Bodhaine et al. (1999)
and Wang (2005) as documented in Section 6.1 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from oceanatmos import rayleigh


class TestRayleighOpticalThickness:
    """Tests for Rayleigh optical thickness calculations."""

    def test_standard_wavelengths(self):
        """Test optical thickness at standard ocean color wavelengths."""
        # Expected values from Bodhaine et al. (1999) Eq. 30
        # at standard pressure 1013.25 hPa
        wavelengths = [412, 443, 490, 510, 555, 670, 765, 865]
        
        for wl in wavelengths:
            tau = rayleigh.rayleigh_optical_thickness(wl)
            assert tau > 0, f"Optical thickness should be positive at {wl} nm"
            assert tau < 1.0, f"Optical thickness should be < 1 at {wl} nm"

    def test_wavelength_dependence(self):
        """Test that optical thickness decreases with wavelength."""
        tau_412 = rayleigh.rayleigh_optical_thickness(412)
        tau_865 = rayleigh.rayleigh_optical_thickness(865)
        
        assert tau_412 > tau_865, "Blue should have higher optical thickness than NIR"

    def test_approximately_lambda_minus_4(self):
        """Test approximate λ⁻⁴ wavelength dependence."""
        tau_443 = rayleigh.rayleigh_optical_thickness(443)
        tau_555 = rayleigh.rayleigh_optical_thickness(555)
        
        # Expected ratio from λ⁻⁴
        expected_ratio = (555 / 443) ** 4
        actual_ratio = tau_443 / tau_555
        
        # Should be close to λ⁻⁴ but not exact due to depolarization
        assert abs(actual_ratio / expected_ratio - 1) < 0.1

    def test_pressure_correction(self):
        """Test pressure correction for optical thickness."""
        tau_standard = rayleigh.rayleigh_optical_thickness(443)
        tau_low_p = rayleigh.rayleigh_optical_thickness(443, pressure=900)
        tau_high_p = rayleigh.rayleigh_optical_thickness(443, pressure=1100)
        
        assert tau_low_p < tau_standard < tau_high_p
        
        # Linear with pressure
        ratio_p = 900 / 1013.25
        expected = tau_standard * ratio_p
        assert abs(tau_low_p - expected) < 0.001

    def test_array_input(self):
        """Test array input for wavelengths."""
        wavelengths = np.array([412, 443, 490, 555])
        tau = rayleigh.rayleigh_optical_thickness(wavelengths)
        
        assert tau.shape == wavelengths.shape
        assert np.all(tau > 0)

    def test_known_value_at_550nm(self):
        """Test against known reference value at 550 nm."""
        # Bodhaine et al. give τ ≈ 0.098 at 550 nm for standard atmosphere
        tau = rayleigh.rayleigh_optical_thickness(550)
        assert abs(tau - 0.098) < 0.005


class TestDepolarizationRatio:
    """Tests for Rayleigh depolarization ratio."""

    def test_positive_values(self):
        """Test that depolarization ratio is positive."""
        wavelengths = [400, 500, 600, 700, 800]
        for wl in wavelengths:
            rho = rayleigh.rayleigh_depolarization_ratio(wl)
            assert rho > 0

    def test_typical_range(self):
        """Test that depolarization ratio is in expected range."""
        # Depolarization ratio for air is typically ~0.03
        rho = rayleigh.rayleigh_depolarization_ratio(550)
        assert 0.02 < rho < 0.05


class TestGeometricAirMassFactor:
    """Tests for geometric air mass factor M."""

    def test_zenith_sun_nadir_view(self):
        """Test M for zenith sun and nadir view."""
        M = rayleigh.geometric_air_mass_factor(0, 0)
        assert M == 2.0  # 1/cos(0) + 1/cos(0) = 2

    def test_symmetric_geometry(self):
        """Test symmetry of air mass factor."""
        M1 = rayleigh.geometric_air_mass_factor(30, 45)
        M2 = rayleigh.geometric_air_mass_factor(45, 30)
        assert abs(M1 - M2) < 1e-10

    def test_increasing_with_angle(self):
        """Test that air mass increases with zenith angle."""
        M1 = rayleigh.geometric_air_mass_factor(0, 0)
        M2 = rayleigh.geometric_air_mass_factor(30, 0)
        M3 = rayleigh.geometric_air_mass_factor(60, 0)
        
        assert M1 < M2 < M3

    def test_array_input(self):
        """Test array input for angles."""
        theta_s = np.array([0, 30, 45, 60])
        theta_v = np.array([0, 0, 0, 0])
        M = rayleigh.geometric_air_mass_factor(theta_s, theta_v)
        
        assert M.shape == theta_s.shape


class TestPressureCorrectionCoefficient:
    """Tests for pressure correction coefficient C(λ, M)."""

    def test_returns_reasonable_values(self):
        """Test that C values are reasonable."""
        C = rayleigh.pressure_correction_coefficient(443, 2.0)
        # C should be a moderate positive value
        assert 0 < C < 2.0

    def test_wavelength_dependence(self):
        """Test wavelength dependence of C."""
        C_blue = rayleigh.pressure_correction_coefficient(412, 2.0)
        C_red = rayleigh.pressure_correction_coefficient(670, 2.0)
        # Different wavelengths should give different C
        assert C_blue != C_red


class TestRayleighReflectancePressureCorrected:
    """Tests for pressure-corrected Rayleigh reflectance."""

    def test_returns_reasonable_values(self):
        """Test that corrected reflectance is reasonable."""
        rho = rayleigh.rayleigh_reflectance_pressure_corrected(
            rho_standard=0.1,
            wavelength=443,
            pressure=1000,
            theta_s=30,
            theta_v=30,
        )
        assert rho > 0
        assert rho < 1.0

    def test_standard_pressure(self):
        """Test that standard pressure gives nearly same value."""
        rho_standard = 0.1
        rho_corrected = rayleigh.rayleigh_reflectance_pressure_corrected(
            rho_standard=rho_standard,
            wavelength=443,
            pressure=1013.25,
            theta_s=30,
            theta_v=30,
        )
        # Should be very close at standard pressure
        assert abs(rho_corrected - rho_standard) < 0.01

    def test_pressure_scaling(self):
        """Test that lower pressure gives lower reflectance."""
        rho_high_p = rayleigh.rayleigh_reflectance_pressure_corrected(
            rho_standard=0.1,
            wavelength=443,
            pressure=1050,
            theta_s=30,
            theta_v=30,
        )
        rho_low_p = rayleigh.rayleigh_reflectance_pressure_corrected(
            rho_standard=0.1,
            wavelength=443,
            pressure=950,
            theta_s=30,
            theta_v=30,
        )
        assert rho_low_p < rho_high_p


class TestRayleighLUT:
    """Tests for Rayleigh look-up table class."""

    def test_initialization(self):
        """Test LUT initialization."""
        lut = rayleigh.RayleighLUT(sensor="seawifs")
        assert lut.sensor == "seawifs"

    def test_interpolate_returns_stokes(self):
        """Test that interpolation returns Stokes-like tuple."""
        lut = rayleigh.RayleighLUT(sensor="seawifs")
        result = lut.interpolate(
            wavelength=443,
            theta_s=30,
            theta_v=30,
            phi=90,
            wind_speed=5.0,
        )
        # Should return (I, Q, U) tuple or similar
        assert len(result) >= 1

    def test_compute_synthetic(self):
        """Test synthetic Rayleigh computation."""
        lut = rayleigh.RayleighLUT(sensor="modis_aqua")
        rho_r = lut.compute_synthetic(
            wavelength=443,
            theta_s=30,
            theta_v=30,
            phi=90,
            wind_speed=5.0,
        )
        assert rho_r > 0
        assert rho_r < 0.5  # Rayleigh reflectance typically < 0.5
