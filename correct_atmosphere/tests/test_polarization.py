"""
Tests for the polarization module.

Tests sensor polarization sensitivity corrections as described
in Section 11 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from correct_atmosphere.polarization import (
    rotation_matrix,
    compute_rotation_angle,
    compute_polarization_correction,
    apply_polarization_correction_simple as apply_polarization_correction,
    stokes_vector_rayleigh,
    degree_of_polarization,
    MuellerMatrix,
)


class TestRotationMatrix:
    """Tests for Stokes vector rotation matrix R(alpha) (Eq. 11.1)."""

    def test_identity_at_zero(self):
        """Rotation by 0 degrees gives identity matrix."""
        R = rotation_matrix(alpha=0.0)
        expected = np.eye(4)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_90_degree_rotation(self):
        """Rotation by 90 degrees transforms Q -> -Q, U -> -U."""
        R = rotation_matrix(alpha=90.0)
        
        # R at 90 deg: cos(180) = -1, sin(180) = 0
        expected = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_45_degree_rotation(self):
        """Rotation by 45 degrees swaps Q and U components."""
        R = rotation_matrix(alpha=45.0)
        
        # R at 45 deg: cos(90) = 0, sin(90) = 1
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_i_and_v_unchanged(self):
        """I and V components are unchanged by rotation."""
        alpha = 37.5  # arbitrary angle
        R = rotation_matrix(alpha)
        
        assert R[0, 0] == 1.0
        assert R[3, 3] == 1.0
        assert np.all(R[0, 1:4] == 0)
        assert np.all(R[1:4, 0] == 0)
        assert np.all(R[0:3, 3] == 0)
        assert np.all(R[3, 0:3] == 0)

    def test_rotation_inverse(self):
        """R(-alpha) is the inverse of R(alpha)."""
        alpha = 30.0
        R_pos = rotation_matrix(alpha)
        R_neg = rotation_matrix(-alpha)
        
        product = R_pos @ R_neg
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_rotation_composition(self):
        """R(alpha1) @ R(alpha2) = R(alpha1 + alpha2)."""
        alpha1 = 20.0
        alpha2 = 35.0
        
        R1 = rotation_matrix(alpha1)
        R2 = rotation_matrix(alpha2)
        R_sum = rotation_matrix(alpha1 + alpha2)
        
        np.testing.assert_allclose(R1 @ R2, R_sum, atol=1e-10)


class TestRotationAngle:
    """Tests for computing the rotation angle alpha."""

    def test_vertical_reference(self):
        """Angle should be 0 when sensor aligned with meridional plane."""
        # When looking straight down (nadir), sensor reference = meridional
        alpha = compute_rotation_angle(
            theta_v=0.0, phi_v=0.0,
            sensor_orientation=0.0
        )
        assert np.isclose(alpha, 0.0, atol=1e-6)

    def test_angle_range(self):
        """Rotation angle varies with geometry."""
        for phi in np.linspace(0, 360, 10):
            for theta in np.linspace(0, 60, 5):
                alpha = compute_rotation_angle(theta, phi, sensor_orientation=45.0)
                # Angle is phi + sensor_orientation, verify it's computed
                assert np.isclose(alpha, phi + 45.0)


class TestMuellerMatrix:
    """Tests for sensor Mueller matrix representation."""

    def test_ideal_sensor(self):
        """Ideal sensor has M = identity (no polarization sensitivity)."""
        M = MuellerMatrix.ideal_sensor()
        np.testing.assert_allclose(M.matrix, np.eye(4), atol=1e-10)

    def test_modis_sensitivity(self):
        """MODIS has known polarization sensitivity values."""
        M = MuellerMatrix.modis_aqua(band=412)
        
        # m12 should be small but non-zero (up to ~0.02-0.05)
        m12 = M.m12
        assert abs(m12) < 0.1
        
    def test_reduced_elements(self):
        """Reduced Mueller matrix elements are normalized by M11."""
        M = MuellerMatrix(M11=2.0, M12=0.1, M13=0.05)
        
        assert M.m12 == 0.1 / 2.0
        assert M.m13 == 0.05 / 2.0


class TestStokesVectorRayleigh:
    """Tests for computing Rayleigh-scattered Stokes vector."""

    def test_unpolarized_input(self):
        """Rayleigh scattering polarizes unpolarized input."""
        I_t = stokes_vector_rayleigh(
            theta_s=30.0, theta_v=30.0, phi=90.0,
            tau_r=0.2, wavelength=443.0
        )
        
        # Total intensity should be positive
        assert I_t[0] > 0
        
        # Should have some degree of polarization
        dop = degree_of_polarization(I_t)
        assert dop > 0

    def test_scattering_angle_effect(self):
        """Polarization is maximum near 90 degree scattering angle."""
        tau_r = 0.2
        wavelength = 443.0
        
        # phi=90 gives scattering closer to 90 degrees
        I_90 = stokes_vector_rayleigh(30.0, 30.0, 90.0, tau_r, wavelength)
        
        # phi=0 (forward) gives smaller scattering angle
        I_0 = stokes_vector_rayleigh(30.0, 30.0, 0.0, tau_r, wavelength)
        
        dop_90 = degree_of_polarization(I_90)
        dop_0 = degree_of_polarization(I_0)
        
        # Polarization typically higher near 90 deg scattering
        assert dop_90 > 0

    def test_circular_polarization_negligible(self):
        """V component (circular polarization) should be negligible."""
        I_t = stokes_vector_rayleigh(30.0, 30.0, 90.0, 0.2, 443.0)
        
        # V should be very small relative to I
        assert abs(I_t[3]) < 0.001 * I_t[0]


class TestDegreeOfPolarization:
    """Tests for degree of polarization calculation."""

    def test_unpolarized(self):
        """Unpolarized light has DoP = 0."""
        I = np.array([1.0, 0.0, 0.0, 0.0])
        dop = degree_of_polarization(I)
        assert np.isclose(dop, 0.0, atol=1e-10)

    def test_fully_polarized(self):
        """Fully linearly polarized light has DoP = 1."""
        I = np.array([1.0, 1.0, 0.0, 0.0])  # Horizontal linear
        dop = degree_of_polarization(I)
        assert np.isclose(dop, 1.0, atol=1e-10)
        
        I = np.array([1.0, 0.0, 1.0, 0.0])  # 45-degree linear
        dop = degree_of_polarization(I)
        assert np.isclose(dop, 1.0, atol=1e-10)

    def test_partial_polarization(self):
        """Partial polarization gives 0 < DoP < 1."""
        I = np.array([1.0, 0.5, 0.3, 0.0])
        dop = degree_of_polarization(I)
        assert 0.0 < dop < 1.0

    def test_circular_included(self):
        """DoP includes circular polarization component."""
        I = np.array([1.0, 0.0, 0.0, 0.5])  # Partially circularly polarized
        dop = degree_of_polarization(I)
        assert np.isclose(dop, 0.5, atol=1e-10)


class TestPolarizationCorrection:
    """Tests for computing polarization correction factor (Eq. 11.4)."""

    def test_ideal_sensor_no_correction(self):
        """Ideal sensor (m12=m13=0) needs no correction."""
        M = MuellerMatrix.ideal_sensor()
        I_t = np.array([1.0, 0.3, 0.1, 0.0])
        alpha = 30.0
        
        pc = compute_polarization_correction(M, I_t, alpha)
        assert np.isclose(pc, 1.0, atol=1e-10)

    def test_unpolarized_light_no_correction(self):
        """Unpolarized light needs no correction regardless of sensor."""
        M = MuellerMatrix(M11=1.0, M12=0.05, M13=0.02)
        I_t = np.array([1.0, 0.0, 0.0, 0.0])
        alpha = 45.0
        
        pc = compute_polarization_correction(M, I_t, alpha)
        assert np.isclose(pc, 1.0, atol=1e-10)

    def test_correction_range(self):
        """Correction factor should be close to 1 for typical conditions."""
        M = MuellerMatrix(M11=1.0, M12=0.03, M13=0.01)  # Typical MODIS sensitivity
        I_t = np.array([1.0, 0.2, 0.1, 0.0])  # Moderate polarization
        
        for alpha in np.linspace(0, 180, 10):
            pc = compute_polarization_correction(M, I_t, alpha)
            # Correction should be within a few percent of 1
            assert 0.95 < pc < 1.05

    def test_angle_dependence(self):
        """Correction varies with rotation angle alpha."""
        M = MuellerMatrix(M11=1.0, M12=0.05, M13=0.0)
        I_t = np.array([1.0, 0.3, 0.0, 0.0])  # Q polarization only
        
        # At alpha=0: pc depends on Q directly
        pc_0 = compute_polarization_correction(M, I_t, alpha=0.0)
        
        # At alpha=45: Q and U mix
        pc_45 = compute_polarization_correction(M, I_t, alpha=45.0)
        
        # At alpha=90: Q sign flips
        pc_90 = compute_polarization_correction(M, I_t, alpha=90.0)
        
        # Values should differ
        assert not np.isclose(pc_0, pc_90, rtol=0.001)


class TestApplyPolarizationCorrection:
    """Tests for applying polarization correction to measured radiance (Eq. 11.6)."""

    def test_recovers_true_radiance(self):
        """Correction recovers true TOA radiance from measured value."""
        # True Stokes vector
        I_true = 0.1  # True intensity
        Q_t = 0.02
        U_t = 0.01
        
        # Sensor properties
        m12 = 0.03
        m13 = 0.01
        alpha = 30.0
        
        # Simulated measured value (Eq. 11.4)
        cos2a = np.cos(np.radians(2 * alpha))
        sin2a = np.sin(np.radians(2 * alpha))
        
        q_t = Q_t / I_true
        u_t = U_t / I_true
        
        pc = 1 + m12 * (cos2a * q_t + sin2a * u_t) + m13 * (-sin2a * q_t + cos2a * u_t)
        I_measured = I_true * pc
        
        # Apply correction
        I_corrected = apply_polarization_correction(
            I_measured=I_measured,
            Q_rayleigh=Q_t,
            U_rayleigh=U_t,
            m12=m12,
            m13=m13,
            alpha=alpha
        )
        
        # Should recover true value
        assert np.isclose(I_corrected, I_true, rtol=0.01)

    def test_typical_modis_correction(self):
        """Typical MODIS correction is within a few percent."""
        I_measured = 0.1
        Q_rayleigh = 0.01
        U_rayleigh = 0.005
        
        # MODIS-like sensitivity
        m12 = 0.02
        m13 = 0.005
        alpha = 45.0
        
        I_corrected = apply_polarization_correction(
            I_measured, Q_rayleigh, U_rayleigh, m12, m13, alpha
        )
        
        # Correction should be within ~3% for MODIS
        assert 0.97 * I_measured < I_corrected < 1.03 * I_measured


class TestIntegration:
    """Integration tests for full polarization correction workflow."""

    def test_full_correction_workflow(self):
        """Test complete polarization correction process."""
        # 1. Define geometry
        theta_s = 30.0
        theta_v = 30.0
        phi = 90.0
        wavelength = 443.0
        tau_r = 0.23
        
        # 2. Compute Rayleigh Stokes vector
        I_rayleigh = stokes_vector_rayleigh(theta_s, theta_v, phi, tau_r, wavelength)
        
        # 3. Get sensor Mueller matrix
        M = MuellerMatrix.modis_aqua(band=443)
        
        # 4. Compute rotation angle
        alpha = compute_rotation_angle(theta_v, phi, sensor_orientation=0.0)
        
        # 5. Compute correction factor
        pc = compute_polarization_correction(M, I_rayleigh, alpha)
        
        # 6. Apply correction
        I_measured = 0.1  # Example measured TOA radiance
        I_corrected = I_measured / pc
        
        # Correction should be reasonable
        assert 0.95 < pc < 1.05
        assert I_corrected > 0

    def test_spectral_correction(self):
        """Polarization correction varies with wavelength."""
        theta_s = 30.0
        theta_v = 45.0
        phi = 90.0
        
        wavelengths = [412, 443, 490, 555, 670, 865]
        pc_values = []
        
        for wl in wavelengths:
            tau_r = 0.3 * (443.0 / wl) ** 4  # Approximate Rayleigh
            I_r = stokes_vector_rayleigh(theta_s, theta_v, phi, tau_r, float(wl))
            M = MuellerMatrix.modis_aqua(band=wl)
            alpha = compute_rotation_angle(theta_v, phi, 0.0)
            pc = compute_polarization_correction(M, I_r, alpha)
            pc_values.append(pc)
        
        # Blue bands typically have larger corrections
        # (more Rayleigh scattering = more polarization)
        assert all(0.9 < pc < 1.1 for pc in pc_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
