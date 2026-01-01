"""
Tests for the normalization module.

Tests normalized reflectances and BRDF corrections as documented in
Section 3 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from oceanatmos import normalization


class TestEarthSunDistanceCorrection:
    """Tests for Earth-Sun distance correction factor."""

    def test_perihelion(self):
        """Test correction at perihelion (early January, day ~3)."""
        # Earth is closest to Sun, so correction factor < 1
        factor = normalization.earth_sun_distance_correction(3)
        assert factor < 1.0
        # Approximately 0.967 (1/1.017²)
        assert 0.96 < factor < 0.98

    def test_aphelion(self):
        """Test correction at aphelion (early July, day ~185)."""
        # Earth is farthest from Sun, so correction factor > 1
        factor = normalization.earth_sun_distance_correction(185)
        assert factor > 1.0
        # Approximately 1.034 (1/0.983²)
        assert 1.02 < factor < 1.05

    def test_annual_variation(self):
        """Test that variation is ~3.4% over year."""
        factors = [normalization.earth_sun_distance_correction(d) 
                   for d in range(1, 366)]
        range_pct = (max(factors) - min(factors)) / np.mean(factors) * 100
        # Should be approximately 6.8% range (±3.4%)
        assert 6 < range_pct < 8


class TestNormalizedWaterLeavingRadiance:
    """Tests for [Lw]_N calculations."""

    def test_formula(self):
        """Test implementation of Eq. 3.2."""
        Lw = 0.5  # W/m²/sr/nm
        theta_s = 30  # degrees
        t_diffuse = 0.9
        day_of_year = 172
        
        Lw_N = normalization.normalized_water_leaving_radiance(
            Lw=Lw,
            theta_s=theta_s,
            t_diffuse=t_diffuse,
            day_of_year=day_of_year,
        )
        
        # Should be > Lw due to normalization
        assert Lw_N > Lw

    def test_solar_zenith_correction(self):
        """Test that zenith sun gives minimum correction."""
        Lw = 0.5
        t_diffuse = 0.9
        
        Lw_N_zenith = normalization.normalized_water_leaving_radiance(
            Lw=Lw, theta_s=0, t_diffuse=t_diffuse, day_of_year=172
        )
        Lw_N_oblique = normalization.normalized_water_leaving_radiance(
            Lw=Lw, theta_s=60, t_diffuse=t_diffuse, day_of_year=172
        )
        
        # Oblique sun requires larger correction
        assert Lw_N_oblique > Lw_N_zenith


class TestNormalizedWaterLeavingReflectance:
    """Tests for [ρw]_N calculations."""

    def test_formula(self):
        """Test implementation of Eq. 3.3."""
        Lw = 0.5  # W/m²/sr/nm
        F0 = 190  # W/m²/nm (typical at 443 nm)
        theta_s = 30
        t_diffuse = 0.9
        
        rho_w_N = normalization.normalized_water_leaving_reflectance(
            Lw=Lw,
            F0=F0,
            theta_s=theta_s,
            t_diffuse=t_diffuse,
            day_of_year=172,
        )
        
        assert 0 < rho_w_N < 1

    def test_pi_factor(self):
        """Test that π factor converts radiance to reflectance."""
        Lw = 1.0
        F0 = 190
        
        Lw_N = normalization.normalized_water_leaving_radiance(
            Lw=Lw, theta_s=0, t_diffuse=1.0, day_of_year=172
        )
        rho_w_N = normalization.normalized_water_leaving_reflectance(
            Lw=Lw, F0=F0, theta_s=0, t_diffuse=1.0, day_of_year=172
        )
        
        # rho = π * L / F0
        expected = np.pi * Lw_N / F0
        assert abs(rho_w_N - expected) < 0.001


class TestRemoteSensingReflectance:
    """Tests for Rrs calculations."""

    def test_definition(self):
        """Test Rrs = Lw / Ed(0+)."""
        Lw = 0.5  # W/m²/sr/nm
        Ed = 100  # W/m²/nm
        
        Rrs = normalization.remote_sensing_reflectance(Lw, Ed)
        expected = Lw / Ed
        
        assert abs(Rrs - expected) < 1e-10

    def test_units(self):
        """Test that Rrs has units of sr⁻¹."""
        # Typical values
        Lw = 0.5  # W/m²/sr/nm
        Ed = 100  # W/m²/nm
        
        Rrs = normalization.remote_sensing_reflectance(Lw, Ed)
        # Typical Rrs values are 0.001-0.01 sr⁻¹
        assert 0.0001 < Rrs < 0.1


class TestRrsToNormalizedReflectance:
    """Tests for converting Rrs to [ρw]_N."""

    def test_pi_factor(self):
        """Test [ρw]_N = π × Rrs."""
        Rrs = 0.005  # sr⁻¹
        rho_w_N = normalization.rrs_to_normalized_reflectance(Rrs)
        
        expected = np.pi * Rrs
        assert abs(rho_w_N - expected) < 1e-10

    def test_array_input(self):
        """Test array input."""
        Rrs = np.array([0.003, 0.005, 0.007, 0.002])
        rho_w_N = normalization.rrs_to_normalized_reflectance(Rrs)
        
        assert rho_w_N.shape == Rrs.shape


class TestSurfaceTransmissionFactorR:
    """Tests for R(θ'v, W) factor."""

    def test_nadir_value(self):
        """Test R0 ≈ 0.529 for nadir viewing."""
        R = normalization.surface_transmission_factor_R(0, wind_speed=0)
        assert abs(R - 0.529) < 0.02

    def test_angle_dependence(self):
        """Test that R varies with viewing angle."""
        R_0 = normalization.surface_transmission_factor_R(0, wind_speed=5)
        R_30 = normalization.surface_transmission_factor_R(30, wind_speed=5)
        R_45 = normalization.surface_transmission_factor_R(45, wind_speed=5)
        
        # R should change with angle
        assert R_0 != R_30 != R_45


class TestFresnelTransmittance:
    """Tests for Fresnel transmittance calculations."""

    def test_normal_incidence(self):
        """Test transmittance at normal incidence."""
        T = normalization.fresnel_transmittance_water_to_air(0)
        # At normal incidence, T ≈ 0.979 for n=1.34
        assert 0.95 < T < 1.0

    def test_total_internal_reflection(self):
        """Test behavior near critical angle."""
        # Critical angle for n=1.34 is about 48°
        T_40 = normalization.fresnel_transmittance_water_to_air(40)
        T_48 = normalization.fresnel_transmittance_water_to_air(48)
        
        assert T_48 < T_40


class TestSnellAngle:
    """Tests for Snell's law calculations."""

    def test_normal_incidence(self):
        """Test that normal incidence remains normal."""
        theta_water = normalization.snell_angle(0, n_water=1.34)
        assert abs(theta_water) < 0.001

    def test_refraction(self):
        """Test refraction from air to water."""
        theta_air = 30
        theta_water = normalization.snell_angle(theta_air, n_water=1.34)
        
        # In-water angle should be smaller
        assert theta_water < theta_air
        
        # Check Snell's law
        expected = np.degrees(np.arcsin(np.sin(np.radians(theta_air)) / 1.34))
        assert abs(theta_water - expected) < 0.01


class TestBRDFCorrection:
    """Tests for BRDF correction class."""

    def test_initialization(self):
        """Test BRDFCorrection initialization."""
        brdf = normalization.BRDFCorrection()
        assert brdf is not None

    def test_get_f_over_Q(self):
        """Test f/Q retrieval."""
        brdf = normalization.BRDFCorrection()
        
        f_Q = brdf.get_f_over_Q(
            wavelength=443,
            chl=0.1,
            theta_s=30,
            theta_v=30,
            phi=90,
            wind_speed=5,
        )
        
        # f/Q typically 0.07-0.15
        assert 0.05 < f_Q < 0.20

    def test_correction_factor(self):
        """Test BRDF correction factor calculation."""
        brdf = normalization.BRDFCorrection()
        
        factor = brdf.correction_factor(
            wavelength=443,
            chl=0.1,
            theta_s=30,
            theta_v=30,
            phi=90,
            wind_speed=5,
        )
        
        # Factor typically 0.6-1.2
        assert 0.5 < factor < 1.5

    def test_nadir_zenith_factor(self):
        """Test that nadir view with zenith sun gives factor ≈ 1."""
        brdf = normalization.BRDFCorrection()
        
        factor = brdf.correction_factor(
            wavelength=443,
            chl=0.1,
            theta_s=0,
            theta_v=0,
            phi=0,
            wind_speed=0,
        )
        
        # Should be close to 1 for reference geometry
        assert 0.9 < factor < 1.1


class TestExactNormalizedReflectance:
    """Tests for [ρw]_N^ex calculations."""

    def test_applies_brdf_correction(self):
        """Test that BRDF correction is applied."""
        Lw = 0.5
        F0 = 190
        theta_s = 45
        theta_v = 30
        
        # Without BRDF
        rho_N = normalization.normalized_water_leaving_reflectance(
            Lw=Lw, F0=F0, theta_s=theta_s, t_diffuse=0.9, day_of_year=172
        )
        
        # With BRDF
        rho_N_ex = normalization.exact_normalized_reflectance(
            Lw=Lw, F0=F0, theta_s=theta_s, theta_v=theta_v, phi=90,
            t_diffuse=0.9, day_of_year=172, chl=0.1, wind_speed=5
        )
        
        # Should be different due to BRDF correction
        assert rho_N != rho_N_ex


class TestNASARrs:
    """Tests for NASA OBPG Rrs product."""

    def test_definition(self):
        """Test Rrs(NASA) = [ρw]_N^ex / π."""
        Lw = 0.5
        F0 = 190
        
        rho_N_ex = normalization.exact_normalized_reflectance(
            Lw=Lw, F0=F0, theta_s=30, theta_v=30, phi=90,
            t_diffuse=0.9, day_of_year=172, chl=0.1, wind_speed=5
        )
        
        Rrs_nasa = normalization.nasa_rrs(
            Lw=Lw, F0=F0, theta_s=30, theta_v=30, phi=90,
            t_diffuse=0.9, day_of_year=172, chl=0.1, wind_speed=5
        )
        
        expected = rho_N_ex / np.pi
        assert abs(Rrs_nasa - expected) < 1e-10

    def test_typical_values(self):
        """Test that Rrs values are in typical range."""
        Rrs = normalization.nasa_rrs(
            Lw=0.5, F0=190, theta_s=30, theta_v=30, phi=90,
            t_diffuse=0.9, day_of_year=172, chl=0.1, wind_speed=5
        )
        
        # Typical Rrs: 0.001-0.01 sr⁻¹
        assert 0.0001 < Rrs < 0.1
