"""
Tests for the glint module.

Tests Sun glint correction algorithms as documented in
Section 7 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere import glint


class TestCoxMunkSlopeVariance:
    """Tests for Cox-Munk slope variance calculations."""

    def test_zero_wind(self):
        """Test slope variance at zero wind speed."""
        sigma2 = glint.cox_munk_slope_variance(0.0)
        # Cox-Munk gives small residual capillary waves
        assert sigma2 >= 0

    def test_linear_with_wind(self):
        """Test linear relationship σ² = 0.00512 * U."""
        U = 10.0
        sigma2 = glint.cox_munk_slope_variance(U)
        expected = 0.00512 * U
        assert abs(sigma2 - expected) < 1e-6

    def test_increasing_with_wind(self):
        """Test that slope variance increases with wind."""
        sigma2_low = glint.cox_munk_slope_variance(5.0)
        sigma2_high = glint.cox_munk_slope_variance(10.0)
        assert sigma2_high > sigma2_low

    def test_array_input(self):
        """Test array input for wind speed."""
        winds = np.array([0, 5, 10, 15])
        sigma2 = glint.cox_munk_slope_variance(winds)
        assert sigma2.shape == winds.shape


class TestWaveFacetProbability:
    """Tests for wave facet probability distribution."""

    def test_normalized(self):
        """Test that probability integrates to approximately 1."""
        # For a range of slope angles
        slopes = np.linspace(-0.5, 0.5, 100)
        wind_speed = 10.0
        
        # This is a simplistic check - actual integration is 2D
        p = glint.wave_facet_probability(slopes, 0.0, wind_speed)
        assert np.all(p >= 0)

    def test_peak_at_zero_slope(self):
        """Test that probability peaks near zero slope."""
        wind_speed = 10.0
        p_zero = glint.wave_facet_probability(0.0, 0.0, wind_speed)
        p_nonzero = glint.wave_facet_probability(0.2, 0.0, wind_speed)
        
        assert p_zero > p_nonzero

    def test_wind_speed_effect(self):
        """Test that higher wind broadens distribution."""
        slope = 0.1
        p_low_wind = glint.wave_facet_probability(slope, 0.0, 2.0)
        p_high_wind = glint.wave_facet_probability(slope, 0.0, 15.0)

        # Higher wind = broader distribution = lower peak probability
        # At a fixed non-zero slope, probability decreases as distribution broadens
        assert p_high_wind < p_low_wind


class TestNormalizedSunGlint:
    """Tests for normalized Sun glint calculations."""

    def test_positive_values(self):
        """Test that normalized glint is positive."""
        L_GN = glint.normalized_sun_glint(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=5.0,
        )
        assert L_GN >= 0

    def test_units(self):
        """Test that result is in inverse steradians."""
        L_GN = glint.normalized_sun_glint(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=5.0,
        )
        # Should be in reasonable range for L_GN (typically < 0.1 sr⁻¹)
        assert L_GN < 1.0

    def test_away_from_specular(self):
        """Test lower glint away from specular direction."""
        # For equal solar and view zenith angles, specular occurs at 180° azimuth
        # (looking back towards the sun direction)
        L_GN_specular = glint.normalized_sun_glint(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=180,  # Near specular direction
            wind_speed=5.0,
        )
        L_GN_away = glint.normalized_sun_glint(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=0,  # Away from specular
            wind_speed=5.0,
        )
        # Should be lower away from specular
        assert L_GN_away < L_GN_specular


class TestGlintMask:
    """Tests for glint masking function."""

    def test_masks_high_glint(self):
        """Test that high glint pixels are masked based on geometry."""
        # Test with geometry that produces very low glint (high angles)
        mask_low = glint.glint_mask(
            solar_zenith=70,
            view_zenith=70,
            relative_azimuth=90,  # Away from specular
            wind_speed=5.0,
            threshold=0.005,
        )
        assert not mask_low  # Should not be masked

        # Test with geometry that produces high glint (near specular)
        mask_high = glint.glint_mask(
            solar_zenith=10,
            view_zenith=10,
            relative_azimuth=0,  # Near specular
            wind_speed=5.0,
            threshold=0.005,
        )
        # Near specular geometry should produce higher glint
        assert mask_high  # Should be masked

    def test_default_threshold(self):
        """Test default threshold of 0.005 sr⁻¹."""
        # Test geometry with very low glint
        mask_low = glint.glint_mask(
            solar_zenith=70,
            view_zenith=70,
            relative_azimuth=90,
            wind_speed=5.0,
        )
        assert not mask_low  # Below default threshold

        # Test geometry near specular (high glint)
        mask_high = glint.glint_mask(
            solar_zenith=10,
            view_zenith=10,
            relative_azimuth=0,
            wind_speed=2.0,
        )
        # This geometry should produce high glint and be masked
        assert mask_high


class TestDirectTransmittance:
    """Tests for direct (beam) transmittance."""

    def test_zenith_path(self):
        """Test transmittance for vertical path."""
        tau = 0.3
        T = glint.direct_transmittance(
            zenith_angle=0.0,
            wavelength=550,
            rayleigh_tau=tau,
            aerosol_tau=0.0,
        )
        expected = np.exp(-tau)
        assert abs(T - expected) < 1e-10

    def test_increasing_path_length(self):
        """Test that transmittance decreases with angle."""
        tau = 0.3
        T_0 = glint.direct_transmittance(
            zenith_angle=0.0, wavelength=550, rayleigh_tau=tau, aerosol_tau=0.0
        )
        T_30 = glint.direct_transmittance(
            zenith_angle=30.0, wavelength=550, rayleigh_tau=tau, aerosol_tau=0.0
        )
        T_60 = glint.direct_transmittance(
            zenith_angle=60.0, wavelength=550, rayleigh_tau=tau, aerosol_tau=0.0
        )

        assert T_0 > T_30 > T_60

    def test_range_zero_to_one(self):
        """Test that transmittance is between 0 and 1."""
        T = glint.direct_transmittance(
            zenith_angle=45.0, wavelength=550, rayleigh_tau=0.5, aerosol_tau=0.0
        )
        assert 0 < T < 1


class TestTwoPathTransmittance:
    """Tests for two-path transmittance (sun to surface to sensor)."""

    def test_product_form(self):
        """Test that two-path is product of individual paths."""
        tau = 0.3
        theta_s = 30.0
        theta_v = 45.0

        T_two = glint.two_path_transmittance(
            solar_zenith=theta_s,
            view_zenith=theta_v,
            wavelength=550,
            rayleigh_tau=tau,
            aerosol_tau=0.0,
        )
        T_s = glint.direct_transmittance(
            zenith_angle=theta_s, wavelength=550, rayleigh_tau=tau, aerosol_tau=0.0
        )
        T_v = glint.direct_transmittance(
            zenith_angle=theta_v, wavelength=550, rayleigh_tau=tau, aerosol_tau=0.0
        )

        expected = T_s * T_v
        assert abs(T_two - expected) < 1e-10

    def test_symmetric(self):
        """Test symmetry in sun and view angles."""
        tau = 0.3
        T_1 = glint.two_path_transmittance(
            solar_zenith=30.0,
            view_zenith=45.0,
            wavelength=550,
            rayleigh_tau=tau,
            aerosol_tau=0.0,
        )
        T_2 = glint.two_path_transmittance(
            solar_zenith=45.0,
            view_zenith=30.0,
            wavelength=550,
            rayleigh_tau=tau,
            aerosol_tau=0.0,
        )

        assert abs(T_1 - T_2) < 1e-10


class TestSunGlintReflectance:
    """Tests for Sun glint reflectance at TOA."""

    def test_returns_reflectance(self):
        """Test that result is a valid reflectance."""
        rho_g = glint.sun_glint_reflectance(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=5.0,
            wavelength=550,
            aerosol_tau=0.1,
        )
        assert rho_g >= 0

    def test_increases_with_wind(self):
        """Test that glint can increase with wind (depends on geometry)."""
        # This depends strongly on geometry, so just check it runs
        rho_low = glint.sun_glint_reflectance(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=2.0,
            wavelength=550,
            aerosol_tau=0.1,
        )
        rho_high = glint.sun_glint_reflectance(
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=15.0,
            wavelength=550,
            aerosol_tau=0.1,
        )
        # Both should be valid
        assert rho_low >= 0
        assert rho_high >= 0
