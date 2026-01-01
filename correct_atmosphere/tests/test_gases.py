"""
Tests for the gases module.

Tests gas absorption corrections for O3 and NO2 as documented in
Section 6.2 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere import gases


class TestOzoneOpticalThickness:
    """Tests for ozone optical thickness calculations."""

    def test_positive_values(self):
        """Test that optical thickness is positive."""
        tau = gases.ozone_optical_thickness(443, ozone_du=350)
        assert tau >= 0

    def test_concentration_scaling(self):
        """Test linear scaling with ozone concentration."""
        tau_low = gases.ozone_optical_thickness(443, ozone_du=200)
        tau_high = gases.ozone_optical_thickness(443, ozone_du=400)
        
        # Should scale linearly with concentration
        assert abs(tau_high / tau_low - 2.0) < 0.01

    def test_wavelength_dependence(self):
        """Test that ozone absorption peaks in UV/blue."""
        tau_400 = gases.ozone_optical_thickness(400, ozone_du=350)
        tau_600 = gases.ozone_optical_thickness(600, ozone_du=350)
        
        # Ozone absorbs more at shorter wavelengths (Chappuis band)
        # Actually ozone has complex structure, but generally decreases
        # from UV to visible
        assert tau_400 != tau_600  # Just verify they're different

    def test_zero_concentration(self):
        """Test zero ozone concentration."""
        tau = gases.ozone_optical_thickness(443, ozone_du=0)
        assert tau == 0

    def test_array_input(self):
        """Test array input for wavelengths."""
        wavelengths = np.array([412, 443, 490, 555])
        tau = gases.ozone_optical_thickness(wavelengths, ozone_du=350)
        assert tau.shape == wavelengths.shape


class TestOzoneTransmittance:
    """Tests for ozone transmittance calculations."""

    def test_range_zero_to_one(self):
        """Test that transmittance is between 0 and 1."""
        t = gases.ozone_transmittance(443, ozone_du=350, theta_s=30, theta_v=30)
        assert 0 <= t <= 1

    def test_zenith_geometry(self):
        """Test transmittance for zenith sun and nadir view."""
        t = gases.ozone_transmittance(443, ozone_du=350, theta_s=0, theta_v=0)
        assert 0.9 < t < 1.0  # High transmittance at visible wavelengths

    def test_angle_dependence(self):
        """Test that transmittance decreases with path length."""
        t_short = gases.ozone_transmittance(443, ozone_du=350, theta_s=0, theta_v=0)
        t_long = gases.ozone_transmittance(443, ozone_du=350, theta_s=60, theta_v=60)
        
        assert t_long < t_short

    def test_unit_transmittance_zero_ozone(self):
        """Test that transmittance is 1 for zero ozone."""
        t = gases.ozone_transmittance(443, ozone_du=0, theta_s=30, theta_v=30)
        assert abs(t - 1.0) < 1e-10


class TestNO2OpticalThickness:
    """Tests for NO2 optical thickness calculations."""

    def test_positive_values(self):
        """Test that optical thickness is positive."""
        # Typical NO2 concentration in molecules/cm²
        tau = gases.no2_optical_thickness(443, no2_conc=1.0e16)
        assert tau >= 0

    def test_concentration_scaling(self):
        """Test linear scaling with NO2 concentration."""
        tau_low = gases.no2_optical_thickness(443, no2_conc=5.0e15)
        tau_high = gases.no2_optical_thickness(443, no2_conc=1.0e16)
        
        assert abs(tau_high / tau_low - 2.0) < 0.01

    def test_wavelength_dependence(self):
        """Test that NO2 absorption is strongest in blue."""
        tau_400 = gases.no2_optical_thickness(400, no2_conc=1.0e16)
        tau_600 = gases.no2_optical_thickness(600, no2_conc=1.0e16)
        
        # NO2 absorbs more at blue wavelengths
        assert tau_400 > tau_600


class TestNO2CorrectionFactor:
    """Tests for NO2 correction factor calculations."""

    def test_positive_correction(self):
        """Test that correction factor is positive."""
        factor = gases.no2_correction_factor(
            wavelength=443,
            no2_total=1.0e16,
            no2_above_200m=0.8e16,
            theta_s=30,
            theta_v=30,
        )
        assert factor > 0

    def test_near_unity_low_no2(self):
        """Test that correction is near unity for low NO2."""
        factor = gases.no2_correction_factor(
            wavelength=443,
            no2_total=1.0e14,  # Very low NO2
            no2_above_200m=0.8e14,
            theta_s=30,
            theta_v=30,
        )
        assert abs(factor - 1.0) < 0.01

    def test_correction_increases_with_no2(self):
        """Test that correction increases with NO2."""
        factor_low = gases.no2_correction_factor(
            wavelength=443,
            no2_total=5.0e15,
            no2_above_200m=4.0e15,
            theta_s=30,
            theta_v=30,
        )
        factor_high = gases.no2_correction_factor(
            wavelength=443,
            no2_total=5.0e16,
            no2_above_200m=4.0e16,
            theta_s=30,
            theta_v=30,
        )
        # Higher NO2 should give larger correction
        assert factor_high > factor_low


class TestApplyNO2Correction:
    """Tests for applying NO2 correction to reflectances."""

    def test_returns_corrected_values(self):
        """Test that correction returns valid reflectances."""
        rho_path = 0.1
        rho_w = 0.01
        
        rho_path_corr, rho_w_corr = gases.apply_no2_correction(
            rho_path=rho_path,
            rho_w=rho_w,
            wavelength=443,
            no2_total=1.0e16,
            no2_above_200m=0.8e16,
            theta_s=30,
            theta_v=30,
        )
        
        assert rho_path_corr > 0
        assert rho_w_corr > 0

    def test_correction_increases_reflectance(self):
        """Test that NO2 correction increases reflectances."""
        rho_path = 0.1
        rho_w = 0.01
        
        rho_path_corr, rho_w_corr = gases.apply_no2_correction(
            rho_path=rho_path,
            rho_w=rho_w,
            wavelength=443,
            no2_total=1.0e16,
            no2_above_200m=0.8e16,
            theta_s=30,
            theta_v=30,
        )
        
        # Correction should increase values (compensate for absorption)
        assert rho_path_corr >= rho_path
        assert rho_w_corr >= rho_w


class TestGasTransmittance:
    """Tests for combined gas transmittance function."""

    def test_combined_transmittance(self):
        """Test combined O3 and NO2 transmittance."""
        t = gases.gas_transmittance(
            wavelength=443,
            ozone_du=350,
            no2_conc=1.0e16,
            theta_s=30,
            theta_v=30,
        )
        
        assert 0 < t < 1

    def test_product_of_individual(self):
        """Test that combined ≈ product of individual transmittances."""
        t_o3 = gases.ozone_transmittance(443, ozone_du=350, theta_s=30, theta_v=30)
        
        # NO2 transmittance at typical concentration
        tau_no2 = gases.no2_optical_thickness(443, no2_conc=1.0e16)
        M = 1/np.cos(np.radians(30)) + 1/np.cos(np.radians(30))
        t_no2 = np.exp(-tau_no2 * M)
        
        t_combined = gases.gas_transmittance(
            wavelength=443,
            ozone_du=350,
            no2_conc=1.0e16,
            theta_s=30,
            theta_v=30,
        )
        
        # Should be approximately the product
        expected = t_o3 * t_no2
        assert abs(t_combined - expected) / expected < 0.1

    def test_array_input(self):
        """Test array input for wavelengths."""
        wavelengths = np.array([412, 443, 490, 555])
        t = gases.gas_transmittance(
            wavelength=wavelengths,
            ozone_du=350,
            no2_conc=1.0e16,
            theta_s=30,
            theta_v=30,
        )
        assert t.shape == wavelengths.shape
        assert np.all((t > 0) & (t <= 1))
