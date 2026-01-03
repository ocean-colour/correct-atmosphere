"""
Tests for the transmittance module.

Tests direct and diffuse atmospheric transmittance calculations
as described in Section 4 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere.transmittance import (
    direct_transmittance,
    diffuse_transmittance_rayleigh,
    diffuse_transmittance,
    two_path_direct_transmittance,
    total_transmittance,
    gaseous_transmittance,
)
from correct_atmosphere.rayleigh import rayleigh_optical_thickness


class TestDirectTransmittance:
    """Tests for direct (beam) transmittance calculations (Section 4.1)."""

    def test_nadir_viewing(self):
        """Direct transmittance at nadir (theta_v = 0) equals exp(-tau)."""
        tau = 0.1
        t_direct = direct_transmittance(zenith_angle=0.0, optical_thickness=tau)
        expected = np.exp(-tau)
        assert np.isclose(t_direct, expected, rtol=1e-10)

    def test_off_nadir_path_length(self):
        """Off-nadir viewing increases path length by 1/cos(theta_v)."""
        tau = 0.1
        theta_v = 60.0  # degrees
        t_direct = direct_transmittance(zenith_angle=theta_v, optical_thickness=tau)
        path_factor = 1.0 / np.cos(np.radians(theta_v))
        expected = np.exp(-tau * path_factor)
        assert np.isclose(t_direct, expected, rtol=1e-10)

    def test_transmittance_decreases_with_tau(self):
        """Transmittance decreases as optical thickness increases."""
        theta_v = 30.0
        tau_values = [0.05, 0.1, 0.2, 0.5]
        t_values = [direct_transmittance(theta_v, tau) for tau in tau_values]
        
        for i in range(len(t_values) - 1):
            assert t_values[i] > t_values[i + 1]

    def test_transmittance_decreases_with_angle(self):
        """Transmittance decreases as viewing angle increases."""
        tau = 0.2
        angles = [0.0, 30.0, 45.0, 60.0]
        t_values = [direct_transmittance(theta_v, tau) for theta_v in angles]
        
        for i in range(len(t_values) - 1):
            assert t_values[i] > t_values[i + 1]

    def test_transmittance_bounds(self):
        """Transmittance must be between 0 and 1."""
        tau = 0.3
        theta_v = 45.0
        t_direct = direct_transmittance(theta_v, tau)
        assert 0.0 < t_direct < 1.0

    def test_zero_optical_thickness(self):
        """Zero optical thickness gives transmittance of 1."""
        t_direct = direct_transmittance(zenith_angle=30.0, optical_thickness=0.0)
        assert np.isclose(t_direct, 1.0, rtol=1e-10)


class TestDiffuseTransmittanceRayleigh:
    """Tests for Rayleigh-only diffuse transmittance."""

    def test_greater_than_direct(self):
        """Diffuse transmittance >= direct transmittance (scattering compensation)."""
        wavelength = 550.0
        theta_v = 30.0
        tau_r = rayleigh_optical_thickness(wavelength)
        
        t_diffuse = diffuse_transmittance_rayleigh(theta_v, wavelength)
        t_direct = direct_transmittance(theta_v, tau_r)
        
        # Diffuse should be >= direct due to forward scattering
        assert t_diffuse >= t_direct

    def test_wavelength_dependence(self):
        """Shorter wavelengths have lower diffuse transmittance."""
        theta_v = 30.0
        wavelengths = [412.0, 555.0, 670.0, 865.0]
        
        t_values = []
        for wl in wavelengths:
            t_values.append(diffuse_transmittance_rayleigh(theta_v, wl))
        
        # Transmittance increases with wavelength (less scattering)
        for i in range(len(t_values) - 1):
            assert t_values[i] < t_values[i + 1]

    def test_typical_values(self):
        """Check typical values are in expected range."""
        theta_v = 30.0
        wavelength = 443.0  # Blue band
        
        t_diffuse = diffuse_transmittance_rayleigh(theta_v, wavelength)
        
        # At 443 nm, tau_r ~ 0.23, expect t ~ 0.85-0.98
        assert 0.80 < t_diffuse < 0.98


class TestDiffuseTransmittance:
    """Tests for combined Rayleigh-aerosol diffuse transmittance."""

    def test_zero_aerosol(self):
        """Zero aerosol optical thickness gives Rayleigh-only transmittance."""
        theta_v = 30.0
        wavelength = 550.0
        
        t = diffuse_transmittance(theta_v, wavelength, aerosol_tau=0.0)
        t_rayleigh = diffuse_transmittance_rayleigh(theta_v, wavelength)
        
        # Should be similar to Rayleigh-only
        assert np.isclose(t, t_rayleigh, rtol=0.01)

    def test_aerosol_reduces_transmittance(self):
        """Adding aerosol reduces transmittance."""
        theta_v = 30.0
        wavelength = 550.0
        
        t_no_aerosol = diffuse_transmittance(theta_v, wavelength, aerosol_tau=0.0)
        t_with_aerosol = diffuse_transmittance(theta_v, wavelength, aerosol_tau=0.2)
        
        assert t_with_aerosol < t_no_aerosol

    def test_reasonable_range(self):
        """Total transmittance in reasonable range for typical conditions."""
        theta_v = 30.0
        wavelength = 490.0
        aerosol_tau = 0.1
        
        t_total = diffuse_transmittance(theta_v, wavelength, aerosol_tau=aerosol_tau)
        
        # Should be between 0.7 and 0.95 for typical conditions
        assert 0.65 < t_total < 0.98


class TestTwoPathTransmittance:
    """Tests for two-path (Sun + view) transmittance (Eq. 7.1)."""

    def test_product_of_paths(self):
        """Two-path transmittance is product of Sun and view path transmittances."""
        tau = 0.3
        theta_s = 30.0
        theta_v = 45.0
        
        t_two = two_path_direct_transmittance(theta_s, theta_v, tau)
        
        # Should be exp(-tau_total * (sec(theta_s) + sec(theta_v)))
        M = 1.0 / np.cos(np.radians(theta_s)) + 1.0 / np.cos(np.radians(theta_v))
        expected = np.exp(-tau * M)
        
        assert np.isclose(t_two, expected, rtol=1e-6)

    def test_symmetric_geometry(self):
        """Symmetric geometry (theta_s = theta_v) gives specific result."""
        tau = 0.3
        theta = 30.0
        
        t_two = two_path_direct_transmittance(theta, theta, tau)
        
        # Should be exp(-2 * tau / cos(theta))
        expected = np.exp(-2.0 * tau / np.cos(np.radians(theta)))
        assert np.isclose(t_two, expected, rtol=1e-6)

    def test_used_for_glint(self):
        """Two-path transmittance appropriate for Sun glint calculations."""
        tau = rayleigh_optical_thickness(550.0) + 0.1  # Rayleigh + aerosol
        theta_s = 20.0
        theta_v = 30.0
        
        t_two = two_path_direct_transmittance(theta_s, theta_v, tau)
        
        # Should be positive and less than 1
        assert 0.0 < t_two < 1.0


class TestTotalTransmittance:
    """Tests for total transmittance function."""

    def test_both_paths(self):
        """Total transmittance with both paths."""
        theta_s = 30.0
        theta_v = 30.0
        wavelength = 550.0
        aerosol_tau = 0.1
        
        t_both = total_transmittance(theta_s, theta_v, wavelength, aerosol_tau, direction='both')
        t_solar = total_transmittance(theta_s, theta_v, wavelength, aerosol_tau, direction='solar')
        t_view = total_transmittance(theta_s, theta_v, wavelength, aerosol_tau, direction='view')
        
        # Both should be approximately product of solar and view
        assert np.isclose(t_both, t_solar * t_view, rtol=0.01)

    def test_solar_path_only(self):
        """Solar path transmittance."""
        theta_s = 30.0
        theta_v = 45.0
        wavelength = 550.0
        
        t_solar = total_transmittance(theta_s, theta_v, wavelength, direction='solar')
        t_direct = diffuse_transmittance(theta_s, wavelength)
        
        assert np.isclose(t_solar, t_direct, rtol=0.01)


class TestGaseousTransmittance:
    """Tests for gaseous absorption transmittance."""

    def test_default_values(self):
        """Gaseous transmittance with default O3 and NO2."""
        theta_s = 30.0
        theta_v = 30.0
        wavelength = 443.0
        
        t_gas = gaseous_transmittance(theta_s, theta_v, wavelength)
        
        # Should be close to 1 (gas absorption is relatively small)
        assert 0.9 < t_gas < 1.0

    def test_wavelength_dependence(self):
        """Gas transmittance varies with wavelength."""
        theta_s = 30.0
        theta_v = 30.0
        
        t_blue = gaseous_transmittance(theta_s, theta_v, 412.0)
        t_red = gaseous_transmittance(theta_s, theta_v, 670.0)
        
        # Both should be positive
        assert t_blue > 0
        assert t_red > 0


class TestIntegration:
    """Integration tests combining transmittance calculations."""

    def test_full_atmospheric_transmittance(self):
        """Calculate full atmospheric transmittance for water-leaving radiance."""
        # Typical conditions
        wavelength = 490.0  # nm
        theta_s = 30.0
        theta_v = 30.0
        tau_a = 0.1
        
        # Diffuse transmittance for water-leaving radiance
        t_diffuse = diffuse_transmittance(theta_v, wavelength, aerosol_tau=tau_a)
        
        # Gaseous transmittance
        t_gas = gaseous_transmittance(theta_s, theta_v, wavelength)
        
        # Total transmittance
        t_total = t_diffuse * t_gas
        
        # Should be in reasonable range (0.7-0.98)
        assert 0.65 < t_total < 0.99

    def test_spectral_transmittance_profile(self):
        """Transmittance increases from blue to NIR wavelengths."""
        wavelengths = [412.0, 443.0, 490.0, 555.0, 670.0, 765.0, 865.0]
        theta_v = 30.0
        tau_a = 0.1
        
        t_values = []
        for wl in wavelengths:
            t = diffuse_transmittance(theta_v, wl, aerosol_tau=tau_a)
            t_values.append(t)
        
        # Generally increasing trend (less scattering at longer wavelengths)
        assert t_values[-1] > t_values[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
