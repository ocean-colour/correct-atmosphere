"""
Tests for the whitecaps module.

Tests whitecap and foam reflectance calculations as documented in
Section 8 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from oceanatmos import whitecaps


class TestWhitecapFraction:
    """Tests for whitecap fractional coverage calculations."""

    def test_zero_below_threshold(self):
        """Test zero whitecap coverage below wind threshold."""
        # Undeveloped seas: threshold is 6.33 m/s
        F = whitecaps.whitecap_fraction_undeveloped(5.0)
        assert F == 0.0

    def test_positive_above_threshold(self):
        """Test positive coverage above threshold."""
        F = whitecaps.whitecap_fraction_undeveloped(10.0)
        assert F > 0

    def test_cubic_dependence(self):
        """Test cubic dependence on wind speed."""
        U1 = 8.0
        U2 = 10.0
        
        F1 = whitecaps.whitecap_fraction_undeveloped(U1)
        F2 = whitecaps.whitecap_fraction_undeveloped(U2)
        
        # F ∝ (U - U0)³
        expected_ratio = ((U2 - 6.33) / (U1 - 6.33)) ** 3
        actual_ratio = F2 / F1
        
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_developed_vs_undeveloped(self):
        """Test that developed seas have more whitecaps."""
        U = 10.0
        F_dev = whitecaps.whitecap_fraction_developed(U)
        F_undev = whitecaps.whitecap_fraction_undeveloped(U)
        
        # Developed seas have lower threshold (4.47 vs 6.33)
        # so more whitecaps at same wind speed
        assert F_dev > F_undev

    def test_reasonable_magnitude(self):
        """Test that coverage is reasonable fraction."""
        F = whitecaps.whitecap_fraction_undeveloped(12.0)
        # Whitecap coverage typically < 5% even at high winds
        assert F < 0.1

    def test_array_input(self):
        """Test array input for wind speed."""
        winds = np.array([5, 7, 9, 11])
        F = whitecaps.whitecap_fraction_undeveloped(winds)
        assert F.shape == winds.shape
        assert F[0] == 0  # Below threshold


class TestWhitecapSpectralFactor:
    """Tests for whitecap spectral reflectance factor."""

    def test_unity_at_blue(self):
        """Test factor is 1.0 at blue wavelengths."""
        a = whitecaps.whitecap_spectral_factor(443)
        assert abs(a - 1.0) < 0.01

    def test_decreases_at_nir(self):
        """Test factor decreases at NIR wavelengths."""
        a_blue = whitecaps.whitecap_spectral_factor(443)
        a_nir = whitecaps.whitecap_spectral_factor(865)
        
        assert a_nir < a_blue

    def test_known_values(self):
        """Test against known values from Frouin et al. (1996)."""
        # From Section 8: awc(865) = 0.645
        a_865 = whitecaps.whitecap_spectral_factor(865)
        assert abs(a_865 - 0.645) < 0.05

    def test_interpolation(self):
        """Test interpolation between known values."""
        a_700 = whitecaps.whitecap_spectral_factor(700)
        # Should be between 555 (1.0) and 765 (0.76)
        assert 0.76 < a_700 < 1.0


class TestWhitecapReflectance:
    """Tests for normalized whitecap reflectance."""

    def test_zero_below_threshold(self):
        """Test zero reflectance below wind threshold."""
        rho = whitecaps.whitecap_reflectance(443, wind_speed=5.0)
        assert rho == 0.0

    def test_positive_above_threshold(self):
        """Test positive reflectance above threshold."""
        rho = whitecaps.whitecap_reflectance(443, wind_speed=10.0)
        assert rho > 0

    def test_formula(self):
        """Test reflectance formula [ρwc]_N = awc × 0.22 × Fwc."""
        U = 10.0
        wavelength = 555
        
        rho = whitecaps.whitecap_reflectance(wavelength, wind_speed=U)
        
        # Calculate expected
        F = whitecaps.whitecap_fraction_undeveloped(U)
        a = whitecaps.whitecap_spectral_factor(wavelength)
        expected = a * 0.22 * F
        
        assert abs(rho - expected) < 1e-6

    def test_spectral_shape(self):
        """Test that reflectance decreases with wavelength."""
        U = 10.0
        rho_443 = whitecaps.whitecap_reflectance(443, wind_speed=U)
        rho_865 = whitecaps.whitecap_reflectance(865, wind_speed=U)
        
        assert rho_865 < rho_443

    def test_max_wind_limit(self):
        """Test behavior at high wind speeds."""
        # Whitecap correction typically applied up to 12 m/s
        rho_12 = whitecaps.whitecap_reflectance(443, wind_speed=12.0)
        rho_15 = whitecaps.whitecap_reflectance(443, wind_speed=15.0)
        
        # Should continue to work at higher winds
        assert rho_15 > rho_12


class TestWhitecapRadiance:
    """Tests for whitecap radiance at sea surface."""

    def test_lambertian(self):
        """Test Lambertian assumption (no directional dependence)."""
        # Whitecap radiance should not depend on viewing direction
        # Just test that it returns valid values
        L = whitecaps.whitecap_radiance(
            wavelength=443,
            wind_speed=10.0,
            F0=190.0,  # W/m²/nm
            theta_s=30,
        )
        assert L >= 0

    def test_scaling_with_irradiance(self):
        """Test linear scaling with solar irradiance."""
        L1 = whitecaps.whitecap_radiance(443, 10.0, F0=100.0, theta_s=30)
        L2 = whitecaps.whitecap_radiance(443, 10.0, F0=200.0, theta_s=30)
        
        assert abs(L2 / L1 - 2.0) < 0.01


class TestWhitecapTOAContribution:
    """Tests for whitecap contribution at TOA."""

    def test_atmospheric_attenuation(self):
        """Test that TOA contribution < surface contribution."""
        # The diffuse transmittance reduces the signal
        rho_toa = whitecaps.whitecap_toa_contribution(
            wavelength=443,
            wind_speed=10.0,
            theta_s=30,
            theta_v=30,
            t_diffuse_s=0.9,
            t_diffuse_v=0.9,
        )
        
        rho_surface = whitecaps.whitecap_reflectance(443, wind_speed=10.0)
        
        # TOA should be less due to atmospheric attenuation
        assert rho_toa < rho_surface

    def test_transmittance_effect(self):
        """Test effect of diffuse transmittance."""
        rho_high_t = whitecaps.whitecap_toa_contribution(
            443, 10.0, 30, 30,
            t_diffuse_s=0.95,
            t_diffuse_v=0.95,
        )
        rho_low_t = whitecaps.whitecap_toa_contribution(
            443, 10.0, 30, 30,
            t_diffuse_s=0.80,
            t_diffuse_v=0.80,
        )
        
        assert rho_high_t > rho_low_t
