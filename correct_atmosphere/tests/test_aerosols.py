"""
Tests for the aerosols module.

Tests aerosol path radiance estimation algorithms as documented in
Section 9 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere import aerosols


class TestAerosolModel:
    """Tests for AerosolModel dataclass."""

    def test_creation(self):
        """Test AerosolModel creation."""
        model = aerosols.AerosolModel(
            model_id=1,
            fine_fraction=0.5,
            angstrom_exponent=1.0,
            effective_radius=0.5,
            relative_humidity=80,
        )
        assert model.model_id == 1
        assert model.fine_fraction == 0.5

    def test_attributes(self):
        """Test that all attributes are accessible."""
        model = aerosols.AerosolModel(
            model_id=5,
            fine_fraction=0.3,
            angstrom_exponent=1.5,
            effective_radius=0.3,
            relative_humidity=70,
        )
        assert hasattr(model, 'angstrom_exponent')
        assert hasattr(model, 'effective_radius')
        assert hasattr(model, 'relative_humidity')


class TestAngstromExponent:
    """Tests for Ångström exponent calculations."""

    def test_from_optical_thickness(self):
        """Test calculation from optical thickness at two wavelengths."""
        # τ(λ) = τ0 (λ0/λ)^α
        # For α = 1.5: τ(443)/τ(865) = (865/443)^1.5
        tau_443 = 0.2
        tau_865 = tau_443 * (443/865)**1.5
        
        alpha = aerosols.angstrom_exponent(tau_443, tau_865, 443, 865)
        assert abs(alpha - 1.5) < 0.01

    def test_small_particles(self):
        """Test that small particles give larger exponent."""
        # Small particles: α typically 1.5-2.0
        # Large particles: α typically 0-0.5
        tau_443 = 0.15
        tau_865_small = tau_443 * (443/865)**1.8  # Small particles
        tau_865_large = tau_443 * (443/865)**0.3  # Large particles
        
        alpha_small = aerosols.angstrom_exponent(tau_443, tau_865_small, 443, 865)
        alpha_large = aerosols.angstrom_exponent(tau_443, tau_865_large, 443, 865)
        
        assert alpha_small > alpha_large

    def test_typical_range(self):
        """Test that exponent is in typical range for marine aerosols."""
        # Marine aerosols: α typically 0.2-1.5
        tau_443 = 0.1
        tau_865 = 0.05  # Typical ratio
        
        alpha = aerosols.angstrom_exponent(tau_443, tau_865, 443, 865)
        assert -0.5 < alpha < 2.5


class TestAerosolOpticalThickness:
    """Tests for aerosol optical thickness calculations."""

    def test_angstrom_law(self):
        """Test Ångström law: τ(λ) = τ0 (λ0/λ)^α."""
        tau_ref = 0.2
        lambda_ref = 865
        alpha = 1.2
        
        tau_443 = aerosols.aerosol_optical_thickness(
            443, tau_ref, lambda_ref, alpha
        )
        
        expected = tau_ref * (lambda_ref / 443) ** alpha
        assert abs(tau_443 - expected) < 1e-6

    def test_reference_wavelength(self):
        """Test that τ equals reference value at reference wavelength."""
        tau_ref = 0.15
        lambda_ref = 865
        
        tau = aerosols.aerosol_optical_thickness(
            lambda_ref, tau_ref, lambda_ref, alpha=1.0
        )
        assert abs(tau - tau_ref) < 1e-10

    def test_wavelength_dependence(self):
        """Test wavelength dependence for positive exponent."""
        tau_ref = 0.1
        alpha = 1.0
        
        tau_blue = aerosols.aerosol_optical_thickness(443, tau_ref, 865, alpha)
        tau_red = aerosols.aerosol_optical_thickness(670, tau_ref, 865, alpha)
        
        assert tau_blue > tau_red  # Blue higher for positive α


class TestEpsilonRatio:
    """Tests for epsilon ratio calculations."""

    def test_definition(self):
        """Test ε = ρA(λ1)/ρA(λ2)."""
        rho_765 = 0.02
        rho_865 = 0.015
        
        epsilon = aerosols.epsilon_ratio(rho_765, rho_865)
        expected = rho_765 / rho_865
        
        assert abs(epsilon - expected) < 1e-10

    def test_typical_range(self):
        """Test epsilon is in typical range."""
        # From Figure 9.3, ε(765,865) typically 0.8-2.5
        rho_765 = 0.02
        rho_865 = 0.015  # Typical values
        
        epsilon = aerosols.epsilon_ratio(rho_765, rho_865)
        assert 0.5 < epsilon < 3.0


class TestAerosolLUT:
    """Tests for aerosol look-up table class."""

    def test_initialization(self):
        """Test LUT initialization."""
        lut = aerosols.AerosolLUT(sensor="seawifs")
        assert lut.sensor == "seawifs"

    def test_get_epsilon(self):
        """Test epsilon retrieval."""
        lut = aerosols.AerosolLUT(sensor="seawifs")
        
        eps = lut.get_epsilon(
            model_index=5,
            wavelength=443,
            ref_wavelength=865,
            theta_s=30,
            theta_v=30,
            phi=90,
        )
        assert eps > 0

    def test_model_range(self):
        """Test that model indices are valid."""
        lut = aerosols.AerosolLUT(sensor="seawifs")
        
        # Should accept model indices 0-9
        for i in range(10):
            eps = lut.get_epsilon(i, 443, 865, 30, 30, 90)
            assert eps > 0


class TestBlackPixelCorrection:
    """Tests for black-pixel atmospheric correction."""

    def test_returns_aerosol_reflectance(self):
        """Test that correction returns aerosol reflectances."""
        rho_toa = {
            412: 0.15,
            443: 0.12,
            490: 0.09,
            555: 0.06,
            670: 0.03,
            765: 0.02,
            865: 0.015,
        }
        
        result = aerosols.black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
            relative_humidity=80,
        )
        
        assert 'rho_a' in result
        assert 'epsilon' in result
        assert 'model_indices' in result

    def test_nir_black_assumption(self):
        """Test that NIR reflectance is attributed to aerosols."""
        rho_toa = {
            765: 0.02,
            865: 0.015,
        }
        
        result = aerosols.black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
        )
        
        # All NIR reflectance should be aerosol under black-pixel assumption
        assert abs(result['rho_a'][865] - rho_toa[865]) < 0.001

    def test_extrapolation_to_visible(self):
        """Test extrapolation to visible wavelengths."""
        rho_toa = {
            412: 0.15,
            443: 0.12,
            765: 0.02,
            865: 0.015,
        }
        
        result = aerosols.black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
        )
        
        # Should estimate aerosol at visible wavelengths
        assert 412 in result['rho_a']
        assert result['rho_a'][412] > 0


class TestEstimateNIRRrs:
    """Tests for NIR Rrs estimation from visible bands."""

    def test_returns_positive(self):
        """Test that estimated NIR Rrs is positive for turbid water."""
        rrs_visible = {
            443: 0.005,
            490: 0.008,
            555: 0.010,
            670: 0.015,  # High 670 indicates turbid water
        }
        
        rrs_nir = aerosols.estimate_nir_rrs(
            rrs_visible=rrs_visible,
            wavelength_nir=765,
        )
        
        assert rrs_nir >= 0

    def test_zero_for_clear_water(self):
        """Test near-zero NIR Rrs for clear Case 1 water."""
        rrs_visible = {
            443: 0.008,
            490: 0.006,
            555: 0.003,
            670: 0.0005,  # Very low red indicates clear water
        }
        
        rrs_765 = aerosols.estimate_nir_rrs(rrs_visible, 765)
        rrs_865 = aerosols.estimate_nir_rrs(rrs_visible, 865)
        
        # Should be very small for clear water
        assert rrs_765 < 0.001
        assert rrs_865 < rrs_765


class TestNonBlackPixelCorrection:
    """Tests for non-black-pixel iterative correction."""

    def test_iterative_convergence(self):
        """Test that iteration converges."""
        rho_toa = {
            412: 0.15,
            443: 0.12,
            490: 0.10,
            555: 0.07,
            670: 0.04,
            765: 0.025,  # Higher NIR suggests non-black water
            865: 0.018,
        }
        
        result = aerosols.non_black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
            max_iterations=10,
            convergence_threshold=0.02,
        )
        
        assert result['converged'] or result['iterations'] == 10
        assert 'rrs_nir' in result

    def test_reduces_aerosol_estimate(self):
        """Test that accounting for water reduces aerosol estimate."""
        rho_toa = {
            412: 0.15,
            443: 0.12,
            490: 0.10,
            555: 0.07,
            670: 0.04,
            765: 0.025,
            865: 0.018,
        }
        
        # Black pixel correction
        bp_result = aerosols.black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
        )
        
        # Non-black pixel correction
        nbp_result = aerosols.non_black_pixel_correction(
            rho_toa=rho_toa,
            sensor="seawifs",
            theta_s=30,
            theta_v=30,
            phi=90,
        )
        
        # Non-black-pixel should give lower aerosol estimate at NIR
        # since some NIR signal is attributed to water
        assert nbp_result['rho_a'][865] <= bp_result['rho_a'][865]


class TestShouldApplyNonBlackPixel:
    """Tests for decision to apply non-black-pixel correction."""

    def test_clear_water(self):
        """Test that clear water doesn't need non-black-pixel."""
        chl = 0.1  # Low chlorophyll
        should_apply = aerosols.should_apply_nonblack_pixel(chl)
        assert not should_apply

    def test_turbid_water(self):
        """Test that turbid water needs non-black-pixel."""
        chl = 1.0  # Higher chlorophyll
        should_apply = aerosols.should_apply_nonblack_pixel(chl)
        assert should_apply

    def test_transition_zone(self):
        """Test transition zone behavior."""
        # 0.3-0.7 mg/m³ is transition zone
        weight = aerosols.should_apply_nonblack_pixel(0.5, return_weight=True)
        assert 0 < weight < 1
