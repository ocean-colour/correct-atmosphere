"""
Tests for the outofband module.

Tests spectral out-of-band response corrections as documented in
Section 10 of NASA TM-2016-217551.
"""

import numpy as np
import pytest

from oceanatmos import outofband


class TestGaussianSRF:
    """Tests for Gaussian spectral response function."""

    def test_peak_at_center(self):
        """Test that SRF peaks at center wavelength."""
        wavelength = np.arange(400, 500, 1)
        center = 443
        fwhm = 10
        
        srf = outofband.gaussian_srf(wavelength, center, fwhm)
        peak_idx = np.argmax(srf)
        
        assert wavelength[peak_idx] == center

    def test_normalized_peak(self):
        """Test that peak value is 1."""
        wavelength = np.arange(400, 500, 1)
        srf = outofband.gaussian_srf(wavelength, 443, 10)
        
        assert abs(np.max(srf) - 1.0) < 0.01

    def test_fwhm(self):
        """Test that width at half maximum matches FWHM."""
        wavelength = np.arange(400, 500, 0.1)
        center = 443
        fwhm = 10
        
        srf = outofband.gaussian_srf(wavelength, center, fwhm)
        
        # Find half-maximum points
        half_max = 0.5
        above_half = wavelength[srf >= half_max]
        actual_fwhm = above_half[-1] - above_half[0]
        
        assert abs(actual_fwhm - fwhm) < 0.5


class TestTophatSRF:
    """Tests for ideal top-hat spectral response function."""

    def test_uniform_response(self):
        """Test uniform response within band."""
        wavelength = np.arange(400, 500, 1)
        center = 443
        width = 10
        
        srf = outofband.tophat_srf(wavelength, center, width)
        
        # Should be 1 within band
        in_band = (wavelength >= 438) & (wavelength <= 448)
        assert np.all(srf[in_band] == 1.0)

    def test_zero_outside(self):
        """Test zero response outside band."""
        wavelength = np.arange(400, 500, 1)
        srf = outofband.tophat_srf(wavelength, 443, 10)
        
        # Should be 0 outside band
        out_band = (wavelength < 438) | (wavelength > 448)
        assert np.all(srf[out_band] == 0.0)


class TestBandAveragedRadiance:
    """Tests for band-averaged radiance calculation."""

    def test_constant_radiance(self):
        """Test with constant radiance spectrum."""
        wavelength = np.arange(400, 500, 1)
        radiance = np.ones_like(wavelength) * 5.0
        srf = outofband.gaussian_srf(wavelength, 443, 10)
        
        L_band = outofband.band_averaged_radiance(wavelength, radiance, srf)
        assert abs(L_band - 5.0) < 0.1

    def test_linear_radiance(self):
        """Test with linear radiance spectrum."""
        wavelength = np.arange(400, 500, 1.0)
        radiance = wavelength.astype(float)  # L = λ
        srf = outofband.tophat_srf(wavelength, 450, 10)
        
        L_band = outofband.band_averaged_radiance(wavelength, radiance, srf)
        # Should be approximately center value for narrow band
        assert abs(L_band - 450) < 1


class TestComputeOOBFractions:
    """Tests for OOB fraction calculations."""

    def test_fractions_sum_to_one(self):
        """Test that in-band + OOB fractions sum to 1."""
        wavelength = np.arange(350, 1000, 1)
        radiance = wavelength.astype(float) ** (-4)  # λ⁻⁴
        srf = outofband.gaussian_srf(wavelength, 443, 10)
        
        f_in, f_low, f_high = outofband.compute_oob_fractions(
            wavelength, radiance, srf,
            lambda_low=438, lambda_high=448
        )
        
        assert abs(f_in + f_low + f_high - 1.0) < 0.01

    def test_ideal_sensor_all_inband(self):
        """Test that ideal sensor has all signal in-band."""
        wavelength = np.arange(400, 500, 1)
        radiance = np.ones_like(wavelength, dtype=float)
        srf = outofband.tophat_srf(wavelength, 443, 10)
        
        f_in, f_low, f_high = outofband.compute_oob_fractions(
            wavelength, radiance, srf,
            lambda_low=438, lambda_high=448
        )
        
        assert f_in > 0.99
        assert f_low < 0.01
        assert f_high < 0.01


class TestCase1RrsSpectrum:
    """Tests for Case 1 water Rrs spectrum model."""

    def test_spectral_shape(self):
        """Test typical Case 1 spectral shape."""
        wavelength = np.arange(400, 700, 1)
        
        # Low chlorophyll
        rrs_low = outofband.case1_rrs_spectrum(wavelength, chl=0.05)
        # High chlorophyll
        rrs_high = outofband.case1_rrs_spectrum(wavelength, chl=5.0)
        
        # Low chl should be bluer (higher 443/555 ratio)
        idx_443 = np.argmin(np.abs(wavelength - 443))
        idx_555 = np.argmin(np.abs(wavelength - 555))
        
        ratio_low = rrs_low[idx_443] / rrs_low[idx_555]
        ratio_high = rrs_high[idx_443] / rrs_high[idx_555]
        
        assert ratio_low > ratio_high

    def test_positive_values(self):
        """Test that Rrs is positive."""
        wavelength = np.arange(400, 800, 1)
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.5)
        
        assert np.all(rrs >= 0)

    def test_typical_magnitude(self):
        """Test typical Rrs magnitude."""
        wavelength = np.array([443, 490, 555])
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.1)
        
        # Typical Rrs: 0.001-0.015 sr⁻¹
        assert np.all(rrs > 0.0005)
        assert np.all(rrs < 0.02)


class TestComputeOOBCorrectionRatio:
    """Tests for OOB correction ratio calculation."""

    def test_near_unity_for_ideal(self):
        """Test ratio near 1 for ideal sensor."""
        wavelength = np.arange(350, 1000, 1)
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.5)
        
        # Ideal top-hat SRF
        srf = outofband.tophat_srf(wavelength, 555, 10)
        
        r = outofband.compute_oob_correction_ratio(
            wavelength, rrs, srf, center=555, ideal_width=10
        )
        
        # Should be very close to 1 for top-hat
        assert abs(r - 1.0) < 0.01

    def test_non_unity_for_real_sensor(self):
        """Test ratio differs from 1 for realistic sensor."""
        wavelength = np.arange(350, 1000, 1)
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.5)
        
        # Gaussian SRF (more realistic, with tails)
        srf = outofband.gaussian_srf(wavelength, 555, 20)  # Wider FWHM
        
        r = outofband.compute_oob_correction_ratio(
            wavelength, rrs, srf, center=555, ideal_width=10
        )
        
        # Should differ from 1 due to OOB response
        assert r != 1.0


class TestOOBCorrectionLUT:
    """Tests for OOB correction look-up table."""

    def test_initialization(self):
        """Test LUT initialization."""
        lut = outofband.OOBCorrectionLUT(sensor="seawifs")
        assert lut.sensor == "seawifs"

    def test_available_sensors(self):
        """Test that expected sensors are available."""
        for sensor in ["seawifs", "modis_aqua", "viirs"]:
            lut = outofband.OOBCorrectionLUT(sensor=sensor)
            assert lut.sensor == sensor

    def test_unknown_sensor_raises(self):
        """Test that unknown sensor raises error."""
        with pytest.raises(ValueError):
            outofband.OOBCorrectionLUT(sensor="unknown_sensor")

    def test_get_correction_factor(self):
        """Test correction factor retrieval."""
        lut = outofband.OOBCorrectionLUT(sensor="seawifs")
        
        # Typical band ratio
        r = lut.get_correction_factor(555, rrs_ratio=1.5)
        
        # Should be close to 1
        assert 0.9 < r < 1.1

    def test_apply_correction(self):
        """Test applying correction to all bands."""
        lut = outofband.OOBCorrectionLUT(sensor="seawifs")
        
        rrs = {
            412: 0.008,
            443: 0.007,
            490: 0.005,
            555: 0.003,
            670: 0.001,
        }
        
        rrs_corr = lut.apply_correction(rrs)
        
        # All bands should be corrected
        assert set(rrs_corr.keys()) == set(rrs.keys())
        
        # Values should be similar (within ~10%)
        for band in rrs:
            if band in rrs_corr:
                ratio = rrs_corr[band] / rrs[band]
                assert 0.85 < ratio < 1.15


class TestApplyOOBCorrection:
    """Tests for convenience function."""

    def test_convenience_function(self):
        """Test that convenience function works."""
        rrs = {412: 0.008, 443: 0.007, 490: 0.005, 555: 0.003}
        
        rrs_corr = outofband.apply_oob_correction(rrs, sensor="seawifs")
        
        assert len(rrs_corr) == len(rrs)


class TestOOBCorrectionForHyperspectral:
    """Tests for hyperspectral data correction."""

    def test_band_convolution(self):
        """Test convolution of hyperspectral data with SRF."""
        wavelength = np.arange(400, 700, 1)
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.3)
        
        rrs_band = outofband.oob_correction_for_hyperspectral(
            wavelength, rrs,
            sensor="seawifs",
            band_wavelength=555
        )
        
        # Should be positive
        assert rrs_band > 0
        
        # Should be close to value at 555 nm
        idx_555 = np.argmin(np.abs(wavelength - 555))
        assert abs(rrs_band - rrs[idx_555]) / rrs[idx_555] < 0.1

    def test_unknown_band_raises(self):
        """Test that unknown band raises error."""
        wavelength = np.arange(400, 700, 1)
        rrs = outofband.case1_rrs_spectrum(wavelength, chl=0.3)
        
        with pytest.raises(ValueError):
            outofband.oob_correction_for_hyperspectral(
                wavelength, rrs,
                sensor="seawifs",
                band_wavelength=999  # Invalid band
            )
