"""
Atmospheric correction of OSOAA-simulated TOA radiances.

Applies the ``correct_atmosphere`` package to TOA reflectances produced by the
OSOAA radiative transfer model, compares with the known "true" water-leaving
reflectance, and quantifies errors.

Key consideration
-----------------
OSOAA is a **scattering-only** model (no gas absorption).  By default the gas
correction step is disabled (ozone = 0, NO2 = 0) so that the correction
assumptions match the simulation.

Example
-------
>>> from dev.correction_on_osoaa_output import run_correction
>>> result, truth, errors = run_correction("nb/osoaa_toa_dataset.json")
>>> print(errors["rmse_vis"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from correct_atmosphere.correction import (
    AncillaryData,
    AtmosphericCorrection,
    CorrectionResult,
    GeometryAngles,
)
from correct_atmosphere.osoaa import load_dataset
from correct_atmosphere import glint


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_osoaa_dataset(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load an OSOAA JSON dataset.

    Delegates to :func:`correct_atmosphere.osoaa.load_dataset` and returns
    the dict with numpy arrays for ``wavelengths_nm``, ``toa_reflectance``
    and ``true_rho_w``.
    """
    return load_dataset(filepath)


def make_geometry(data: Dict[str, Any]) -> GeometryAngles:
    """Build a :class:`GeometryAngles` from a loaded OSOAA dataset dict."""
    g = data["geometry"]
    return GeometryAngles(
        solar_zenith=g["solar_zenith"],
        solar_azimuth=g["solar_azimuth"],
        view_zenith=g["view_zenith"],
        view_azimuth=g["view_azimuth"],
    )


def make_ancillary(
    data: Dict[str, Any],
    disable_gas: bool = True,
) -> AncillaryData:
    """Build an :class:`AncillaryData` from a loaded OSOAA dataset dict.

    Parameters
    ----------
    data : dict
        Loaded OSOAA dataset (from :func:`load_osoaa_dataset`).
    disable_gas : bool
        If True (default), set ozone, water_vapor, and NO2 to zero so
        that the correction matches OSOAA's scattering-only physics.
    """
    a = data["ancillary"]
    if disable_gas:
        return AncillaryData(
            pressure=a["pressure"],
            wind_speed=a["wind_speed"],
            ozone=0.0,
            water_vapor=0.0,
            relative_humidity=a["relative_humidity"],
            no2_total=0.0,
            no2_stratospheric=0.0,
        )
    return AncillaryData(
        pressure=a["pressure"],
        wind_speed=a["wind_speed"],
        ozone=a["ozone"],
        water_vapor=a.get("water_vapor", 1.5),
        relative_humidity=a["relative_humidity"],
    )


def osoaa_reflectance_to_radiance(
    toa_refl: np.ndarray,
    F0: np.ndarray,
) -> np.ndarray:
    """Convert OSOAA normalised reflectance to radiance.

    OSOAA output is  rho = pi * L / F0.
    The correction processor expects  L  in mW cm-2 um-1 sr-1.

    Parameters
    ----------
    toa_refl : ndarray
        OSOAA TOA reflectance (pi * Lt / F0).
    F0 : ndarray
        Extra-terrestrial solar irradiance per band (mW cm-2 um-1).

    Returns
    -------
    ndarray
        TOA radiance Lt (mW cm-2 um-1 sr-1).
    """
    return toa_refl * F0 / np.pi


# ---------------------------------------------------------------------------
# Correction pipeline
# ---------------------------------------------------------------------------

def run_correction(
    dataset_path: Union[str, Path],
    sensor: str = "modis_aqua",
    disable_gas: bool = True,
) -> tuple[CorrectionResult, np.ndarray, Dict[str, Any]]:
    """Load an OSOAA dataset, run the atmospheric correction, and score it.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the ``osoaa_toa_dataset.json`` file.
    sensor : str
        Sensor name passed to :class:`AtmosphericCorrection`.
    disable_gas : bool
        Disable gas correction to match OSOAA assumptions (default True).

    Returns
    -------
    result : CorrectionResult
        Full atmospheric correction output.
    true_rho_w : ndarray
        OSOAA "truth" water-leaving reflectance.
    errors : dict
        Error statistics (see :func:`compute_error_stats`).
    """
    data = load_osoaa_dataset(dataset_path)
    ac = AtmosphericCorrection(sensor)
    F0 = ac._get_solar_irradiance()

    Lt = osoaa_reflectance_to_radiance(data["toa_reflectance"], F0)
    geometry = make_geometry(data)
    ancillary = make_ancillary(data, disable_gas=disable_gas)

    result = ac.process(Lt, geometry, ancillary)
    errors = compute_error_stats(
        result.rho_w, data["true_rho_w"], data["wavelengths_nm"],
    )
    return result, data["true_rho_w"], errors


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_stats(
    retrieved_rho_w: np.ndarray,
    true_rho_w: np.ndarray,
    wavelengths: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-band and summary error statistics.

    Parameters
    ----------
    retrieved_rho_w : ndarray
        Retrieved water-leaving reflectance from the correction.
    true_rho_w : ndarray
        OSOAA truth water-leaving reflectance.
    wavelengths : ndarray
        Wavelengths (nm) for each band.

    Returns
    -------
    dict
        Keys:

        - ``diff`` : per-band difference (retrieved - true)
        - ``rel_err`` : per-band relative error (%)
        - ``mae`` : mean absolute error (all bands)
        - ``rmse`` : root-mean-square error (all bands)
        - ``mae_vis`` : mean absolute error (visible bands < 700 nm)
        - ``rmse_vis`` : RMSE (visible bands)
        - ``mean_rel_err_vis`` : mean relative error in visible (%)
        - ``vis_mask`` : boolean mask for visible bands
    """
    diff = retrieved_rho_w - true_rho_w
    # Guard against near-zero truth in NIR
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(true_rho_w) > 1e-6,
            100.0 * diff / true_rho_w,
            np.nan,
        )

    vis = wavelengths < 700
    vis_diff = diff[vis]

    return {
        "wavelengths": wavelengths,
        "diff": diff,
        "rel_err": rel_err,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae_vis": float(np.mean(np.abs(vis_diff))),
        "rmse_vis": float(np.sqrt(np.mean(vis_diff ** 2))),
        "mean_rel_err_vis": float(np.nanmean(rel_err[vis])),
        "vis_mask": vis,
    }


# ---------------------------------------------------------------------------
# Correction breakdown
# ---------------------------------------------------------------------------

@dataclass
class CorrectionBreakdown:
    """Individual reflectance components removed during correction."""

    wavelengths: np.ndarray
    rho_t: np.ndarray     # TOA reflectance (internal convention)
    rho_r: np.ndarray     # Rayleigh
    rho_g: np.ndarray     # Sun glint
    rho_wc: np.ndarray    # Whitecap
    rho_A: np.ndarray     # Aerosol
    rho_w: np.ndarray     # Retrieved water-leaving
    t_dv: np.ndarray      # Diffuse transmittance


def correction_breakdown(
    ac: AtmosphericCorrection,
    geometry: GeometryAngles,
    ancillary: AncillaryData,
    toa_refl: np.ndarray,
    result: CorrectionResult,
) -> CorrectionBreakdown:
    """Re-run internal AC steps to extract individual components.

    Parameters
    ----------
    ac : AtmosphericCorrection
        The processor instance (already used for the correction).
    geometry : GeometryAngles
        Observation geometry.
    ancillary : AncillaryData
        Ancillary data used in the correction.
    toa_refl : ndarray
        OSOAA TOA reflectance (pi * Lt / F0).
    result : CorrectionResult
        The correction result (used for La → rho_A conversion).

    Returns
    -------
    CorrectionBreakdown
    """
    cos_sza = np.cos(geometry.solar_zenith_rad)
    F0 = ac._get_solar_irradiance()

    # Internal TOA reflectance convention: rho_t = pi*Lt / (F0*cos_sza)
    rho_t = toa_refl / cos_sza

    rho_r = ac._compute_rayleigh_reflectance(geometry, ancillary)
    rho_wc = ac._compute_whitecap_reflectance(ancillary.wind_speed)
    t_dv = ac._compute_diffuse_transmittance(geometry, ancillary)

    # Glint reflectance per band
    _vec_glint = np.vectorize(glint.sun_glint_reflectance)
    rho_g = np.zeros(len(ac.wavelengths))
    for i, wl in enumerate(ac.wavelengths):
        rho_g[i] = _vec_glint(
            geometry.solar_zenith,
            geometry.view_zenith,
            geometry.relative_azimuth,
            ancillary.wind_speed,
            wl,
        )

    # Aerosol reflectance back-derived from result.La
    rho_A = result.La * np.pi / (F0 * cos_sza)

    return CorrectionBreakdown(
        wavelengths=ac.wavelengths,
        rho_t=rho_t,
        rho_r=rho_r,
        rho_g=rho_g,
        rho_wc=rho_wc,
        rho_A=rho_A,
        rho_w=result.rho_w,
        t_dv=t_dv,
    )


# ---------------------------------------------------------------------------
# Gas-impact comparison
# ---------------------------------------------------------------------------

def compare_gas_impact(
    dataset_path: Union[str, Path],
    sensor: str = "modis_aqua",
) -> Dict[str, Any]:
    """Run correction with and without gas absorption and compare.

    Returns
    -------
    dict
        Keys: ``result_no_gas``, ``result_with_gas``, ``true_rho_w``,
        ``errors_no_gas``, ``errors_with_gas``, ``wavelengths``.
    """
    data = load_osoaa_dataset(dataset_path)
    ac = AtmosphericCorrection(sensor)
    F0 = ac._get_solar_irradiance()
    Lt = osoaa_reflectance_to_radiance(data["toa_reflectance"], F0)
    geometry = make_geometry(data)

    anc_no_gas = make_ancillary(data, disable_gas=True)
    anc_with_gas = make_ancillary(data, disable_gas=False)

    result_no_gas = ac.process(Lt, geometry, anc_no_gas)
    result_with_gas = ac.process(Lt, geometry, anc_with_gas)

    true_rho_w = data["true_rho_w"]
    wl = data["wavelengths_nm"]

    return {
        "result_no_gas": result_no_gas,
        "result_with_gas": result_with_gas,
        "true_rho_w": true_rho_w,
        "errors_no_gas": compute_error_stats(result_no_gas.rho_w, true_rho_w, wl),
        "errors_with_gas": compute_error_stats(result_with_gas.rho_w, true_rho_w, wl),
        "wavelengths": wl,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_summary(
    result: CorrectionResult,
    true_rho_w: np.ndarray,
    errors: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a human-readable comparison table."""
    wl = errors["wavelengths"]

    if scenario:
        print(f"Scenario: {scenario.get('name', '?')}")
        print(f"  Chlorophyll (true): {scenario.get('chlorophyll', '?')} mg/m³")
        print(f"  AOT(550):           {scenario.get('aot_550', '?')}")
        print()

    print(f"Estimated chlorophyll: {result.chlorophyll:.3f} mg/m³")
    print()
    hdr = f"{'Band (nm)':<12} {'True':>10} {'Retrieved':>12} {'Diff':>10} {'Rel Err':>10}"
    print(hdr)
    print("-" * len(hdr))
    for i, w in enumerate(wl):
        d = errors["diff"][i]
        re = errors["rel_err"][i]
        re_str = f"{re:.1f}%" if np.isfinite(re) else "  n/a"
        print(f"{w:>8.0f}     {true_rho_w[i]:>10.6f} "
              f"{result.rho_w[i]:>12.6f} {d:>10.6f} {re_str:>10}")

    print()
    print(f"RMSE (visible): {errors['rmse_vis']:.6f}")
    print(f"MAE  (visible): {errors['mae_vis']:.6f}")
    print(f"Mean relative error (visible): {errors['mean_rel_err_vis']:.1f}%")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(dataset_path: str = "nb/osoaa_toa_dataset.json") -> None:
    """Run the full correction pipeline and print results."""
    data = load_osoaa_dataset(dataset_path)
    result, true_rho_w, errors = run_correction(dataset_path)
    print_summary(result, true_rho_w, errors, scenario=data.get("scenario"))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "nb/osoaa_toa_dataset.json"
    main(path)
