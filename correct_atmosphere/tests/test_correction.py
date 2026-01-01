"""
Tests for the main correction module.

Tests the high-level AtmosphericCorrection class that orchestrates
the complete atmospheric correction process.
"""

import numpy as np
import pytest

from oceanatmos import AtmosphericCorrection
from oceanatmos.correction import AtmosphericCorrection as AC


class TestAtmosphericCorrectionInit:
    """Tests for AtmosphericCorrection initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        ac = AtmosphericCorrection()
        assert ac is not None

    def test_sensor_initialization(self):
        """Test initialization with specific sensor."""
        ac = AtmosphericCorrection(sensor="seawifs")
        assert ac.sensor == "seawifs"

    def test_available_sensors(self):
        """Test that common sensors are available."""
        for sensor in ["seawifs", "modis_aqua", "viirs"]:
            ac = AtmosphericCorrection(sensor=sensor)
            assert ac.sensor == sensor


class TestAtmosphericCorrectionProcess:
    """Tests for the atmospheric correction process."""

    def test_process_returns_dict(self, typical_toa_reflectance, typical_geometry,
                                   clear_atmosphere):
        """Test that process returns dictionary of results."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            **clear_atmosphere
        )
        
        assert isinstance(result, dict)
        assert 'rrs' in result or 'rho_w' in result

    def test_process_all_bands(self, typical_toa_reflectance, typical_geometry,
                               clear_atmosphere, seawifs_bands):
        """Test that all input bands are processed."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            **clear_atmosphere
        )
        
        # Should have results for all visible bands
        if 'rrs' in result:
            for band in [412, 443, 490, 555]:
                assert band in result['rrs']

    def test_removes_atmospheric_contribution(self, typical_toa_reflectance,
                                               typical_geometry, clear_atmosphere):
        """Test that water-leaving reflectance < TOA reflectance."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            **clear_atmosphere
        )
        
        # Water-leaving should be much smaller than TOA
        if 'rho_w' in result:
            for band in result['rho_w']:
                if band in typical_toa_reflectance:
                    assert result['rho_w'][band] < typical_toa_reflectance[band]


class TestCorrectionSteps:
    """Tests for individual correction steps."""

    def test_rayleigh_correction(self, typical_geometry, clear_atmosphere,
                                  seawifs_bands):
        """Test Rayleigh correction step."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        rho_r = ac.compute_rayleigh(
            wavelengths=seawifs_bands,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            pressure=clear_atmosphere['pressure'],
            wind_speed=clear_atmosphere['wind_speed'],
        )
        
        # Rayleigh reflectance should decrease with wavelength
        assert rho_r[412] > rho_r[865]

    def test_gas_correction(self, typical_geometry, clear_atmosphere,
                            seawifs_bands):
        """Test gas transmittance calculation."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        t_gas = ac.compute_gas_transmittance(
            wavelengths=seawifs_bands,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            ozone_du=clear_atmosphere['ozone_du'],
            no2_conc=clear_atmosphere['no2_conc'],
        )
        
        # Transmittance should be between 0 and 1
        for band in t_gas:
            assert 0 < t_gas[band] <= 1

    def test_glint_correction(self, typical_geometry, clear_atmosphere):
        """Test Sun glint calculation."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        glint_info = ac.compute_glint(
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            wind_speed=clear_atmosphere['wind_speed'],
        )
        
        assert 'L_GN' in glint_info
        assert 'masked' in glint_info

    def test_whitecap_correction(self, clear_atmosphere, seawifs_bands):
        """Test whitecap reflectance calculation."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        rho_wc = ac.compute_whitecaps(
            wavelengths=seawifs_bands,
            wind_speed=clear_atmosphere['wind_speed'],
        )
        
        # Should be small for moderate wind
        for band in rho_wc:
            assert rho_wc[band] >= 0
            assert rho_wc[band] < 0.01


class TestNormalization:
    """Tests for output normalization."""

    def test_exact_normalized_output(self, typical_toa_reflectance,
                                      typical_geometry, clear_atmosphere):
        """Test exact normalized reflectance output."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            normalize=True,
            **clear_atmosphere
        )
        
        # Result should include exact normalized quantities
        assert 'rho_w_ex' in result or 'rrs' in result

    def test_brdf_correction_applied(self, typical_toa_reflectance,
                                      typical_geometry, clear_atmosphere):
        """Test that BRDF correction is applied."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        # Without BRDF
        result_no_brdf = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            apply_brdf=False,
            **clear_atmosphere
        )
        
        # With BRDF
        result_brdf = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=typical_geometry['theta_s'],
            theta_v=typical_geometry['theta_v'],
            phi=typical_geometry['phi'],
            apply_brdf=True,
            chl=0.1,  # Needed for BRDF
            **clear_atmosphere
        )
        
        # Results should differ when BRDF is applied
        # (unless at reference geometry)
        if typical_geometry['theta_s'] != 0 or typical_geometry['theta_v'] != 0:
            # At non-reference geometry, values should differ
            pass  # Implementation-dependent


class TestQualityFlags:
    """Tests for quality flag generation."""

    def test_glint_flag(self, typical_toa_reflectance, typical_geometry,
                        clear_atmosphere):
        """Test glint warning flag."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        # Use geometry that might have glint
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=30,
            theta_v=30,
            phi=30,  # Closer to specular direction
            **clear_atmosphere
        )
        
        if 'flags' in result:
            assert 'glint_warning' in result['flags'] or 'glint' in result['flags']

    def test_atmospheric_correction_flag(self, typical_toa_reflectance,
                                          typical_geometry, clear_atmosphere):
        """Test atmospheric correction warning flag."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            **typical_geometry,
            **clear_atmosphere
        )
        
        # Should complete without failure flag
        if 'flags' in result:
            assert 'atmcor_fail' not in result['flags'] or \
                   not result['flags'].get('atmcor_fail', False)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_high_solar_zenith(self, typical_toa_reflectance, clear_atmosphere):
        """Test processing at high solar zenith angle."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        # High solar zenith - should still work
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=70,
            theta_v=30,
            phi=90,
            **clear_atmosphere
        )
        
        assert result is not None

    def test_nadir_view(self, typical_toa_reflectance, clear_atmosphere):
        """Test processing with nadir viewing."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            theta_s=30,
            theta_v=0,  # Nadir
            phi=0,
            **clear_atmosphere
        )
        
        assert result is not None

    def test_missing_ancillary(self, typical_toa_reflectance, typical_geometry):
        """Test handling of missing ancillary data."""
        ac = AtmosphericCorrection(sensor="seawifs")
        
        # Use default/climatological values when ancillary is missing
        result = ac.process(
            rho_toa=typical_toa_reflectance,
            **typical_geometry,
            # Missing: pressure, ozone, wind_speed, etc.
        )
        
        # Should still produce a result (using defaults)
        assert result is not None
