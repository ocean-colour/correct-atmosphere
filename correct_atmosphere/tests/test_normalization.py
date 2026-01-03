"""
Tests for the normalization module.

Tests normalized reflectances and BRDF corrections as documented in
Section 3 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere import normalization


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
        lw = 0.5  # W/m²/sr/nm
        solar_zenith = 30  # degrees
        diffuse_transmittance = 0.9
        day_of_year = 172

        lw_n = normalization.normalized_water_leaving_radiance(
            lw=lw,
            solar_zenith=solar_zenith,
            diffuse_transmittance=diffuse_transmittance,
            day_of_year=day_of_year,
        )

        # Should be > lw due to normalization
        assert lw_n > lw

    def test_solar_zenith_correction(self):
        """Test that zenith sun gives minimum correction."""
        lw = 0.5
        diffuse_transmittance = 0.9

        lw_n_zenith = normalization.normalized_water_leaving_radiance(
            lw=lw, solar_zenith=0, diffuse_transmittance=diffuse_transmittance, day_of_year=172
        )
        lw_n_oblique = normalization.normalized_water_leaving_radiance(
            lw=lw, solar_zenith=60, diffuse_transmittance=diffuse_transmittance, day_of_year=172
        )

        # Oblique sun requires larger correction
        assert lw_n_oblique > lw_n_zenith


class TestNormalizedWaterLeavingReflectance:
    """Tests for [ρw]_N calculations."""

    def test_formula(self):
        """Test implementation of Eq. 3.3."""
        lw = 0.5  # W/m²/sr/nm
        solar_irradiance = 190  # W/m²/nm (typical at 443 nm)
        solar_zenith = 30
        diffuse_transmittance = 0.9

        rho_w_n = normalization.normalized_water_leaving_reflectance(
            lw=lw,
            solar_irradiance=solar_irradiance,
            solar_zenith=solar_zenith,
            diffuse_transmittance=diffuse_transmittance,
            day_of_year=172,
        )

        assert 0 < rho_w_n < 1

    def test_pi_factor(self):
        """Test that π factor converts radiance to reflectance."""
        lw = 1.0
        solar_irradiance = 190

        lw_n = normalization.normalized_water_leaving_radiance(
            lw=lw, solar_zenith=0, diffuse_transmittance=1.0, day_of_year=172
        )
        rho_w_n = normalization.normalized_water_leaving_reflectance(
            lw=lw, solar_irradiance=solar_irradiance, solar_zenith=0,
            diffuse_transmittance=1.0, day_of_year=172
        )

        # rho = π * L / F0
        expected = np.pi * lw_n / solar_irradiance
        assert abs(rho_w_n - expected) < 0.001


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
        # At normal incidence, T is reduced by solid angle change factor
        # T = (1 - reflectance) * (cos_air / cos_water) / n^2
        # For normal incidence: T ≈ 0.979 / 1.34^2 ≈ 0.545
        assert 0.4 < T < 0.7

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
            chlorophyll=0.1,
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=5,
        )

        # f/Q typically 0.07-0.15
        assert 0.05 < f_Q < 0.20

    def test_correction_factor(self):
        """Test BRDF correction factor calculation."""
        brdf = normalization.BRDFCorrection()

        factor = brdf.correction_factor(
            wavelength=443,
            chlorophyll=0.1,
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            wind_speed=5,
        )

        # Factor typically 0.6-1.2
        assert 0.5 < factor < 1.5

    def test_nadir_zenith_factor(self):
        """Test that nadir view with zenith sun gives factor ≈ 1."""
        brdf = normalization.BRDFCorrection()

        factor = brdf.correction_factor(
            wavelength=443,
            chlorophyll=0.1,
            solar_zenith=0,
            view_zenith=0,
            relative_azimuth=0,
            wind_speed=0,
        )

        # Should be close to 1 for reference geometry
        assert 0.9 < factor < 1.1


class TestExactNormalizedReflectance:
    """Tests for [ρw]_N^ex calculations."""

    def test_applies_brdf_correction(self):
        """Test that BRDF correction is applied."""
        rho_w = 0.01  # Water-leaving reflectance
        solar_irradiance = 190
        solar_zenith = 45
        view_zenith = 30

        # Without BRDF - use basic normalization
        rho_n = normalization.normalized_water_leaving_reflectance(
            lw=rho_w * solar_irradiance / np.pi,  # Convert reflectance to radiance
            solar_irradiance=solar_irradiance,
            solar_zenith=solar_zenith,
            diffuse_transmittance=0.9,
            day_of_year=172
        )

        # With BRDF
        rho_n_ex = normalization.exact_normalized_reflectance(
            rho_w=rho_w,
            wavelength=443,
            solar_zenith=solar_zenith,
            view_zenith=view_zenith,
            relative_azimuth=90,
            chlorophyll=0.1,
            solar_irradiance=solar_irradiance,
            diffuse_transmittance=0.9,
            day_of_year=172,
            wind_speed=5,
        )

        # Should be different due to BRDF correction
        assert rho_n != rho_n_ex


class TestNASARrs:
    """Tests for NASA OBPG Rrs product."""

    def test_definition(self):
        """Test Rrs(NASA) = [ρw]_N^ex / π."""
        rho_w = 0.01  # Water-leaving reflectance
        solar_irradiance = 190

        rho_n_ex = normalization.exact_normalized_reflectance(
            rho_w=rho_w,
            wavelength=443,
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            chlorophyll=0.1,
            solar_irradiance=solar_irradiance,
            diffuse_transmittance=0.9,
            day_of_year=172,
            wind_speed=5,
        )

        # nasa_rrs just takes the exact normalized reflectance and divides by pi
        rrs_nasa = normalization.nasa_rrs(rho_n_ex)

        expected = rho_n_ex / np.pi
        assert abs(rrs_nasa - expected) < 1e-10

    def test_typical_values(self):
        """Test that Rrs values are in typical range."""
        rho_w = 0.01  # Water-leaving reflectance
        solar_irradiance = 190

        rho_n_ex = normalization.exact_normalized_reflectance(
            rho_w=rho_w,
            wavelength=443,
            solar_zenith=30,
            view_zenith=30,
            relative_azimuth=90,
            chlorophyll=0.1,
            solar_irradiance=solar_irradiance,
            diffuse_transmittance=0.9,
            day_of_year=172,
            wind_speed=5,
        )

        rrs = normalization.nasa_rrs(rho_n_ex)

        # Typical Rrs: 0.001-0.01 sr⁻¹
        assert 0.0001 < rrs < 0.1
