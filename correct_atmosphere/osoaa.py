"""
OSOAA TOA simulation wrapper for atmospheric correction testing.

This module wraps the OSOAA (Ocean Successive Orders with Atmosphere - Advanced)
Fortran radiative transfer code, providing Python classes and functions to:

- Configure and run TOA and subsurface radiance simulations
- Parse standard (LUM_vsVZA) and advanced (LUM_Advanced_Up) output files
- Extract radiance components at key atmospheric levels
- Package results into datasets for atmospheric correction validation
- Save/load datasets as JSON

OSOAA conventions:
    - Wavelength is in **micrometers** internally (this module accepts nm)
    - Imaginary refractive indices are **negative** (absorption convention)
    - Output radiances are normalised: pi * L / E_sun

Example
-------
>>> from correct_atmosphere.osoaa import OSOAASimulation, SimulationScenario
>>> sim = OSOAASimulation("/path/to/OSOAA")
>>> scenario = SimulationScenario("test", 30.0, 15.0, 90.0, 0.3, 0.1, 5.0)
>>> params = sim.get_toa_params(wavelength_nm=550.0, solar_zenith=30.0)
>>> results = sim.run(params)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from correct_atmosphere.correction import AncillaryData, GeometryAngles

from IPython import embed

__all__ = [
    "OSOAASimulation",
    "SimulationScenario",
    "OSOAADataset",
    "scale_aot_angstrom",
    "create_dataset",
    "save_dataset",
    "load_dataset",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SimulationScenario:
    """Container for OSOAA simulation parameters."""

    name: str
    solar_zenith: float
    view_zenith: float
    relative_azimuth: float
    chlorophyll: float
    aot_550: float
    wind_speed: float
    pressure: float = 1013.25
    ozone: float = 300.0

    def __repr__(self) -> str:
        return (
            f"{self.name}: SZA={self.solar_zenith}\u00b0, "
            f"VZA={self.view_zenith}\u00b0, "
            f"RAA={self.relative_azimuth}\u00b0, "
            f"Chl={self.chlorophyll} mg/m\u00b3, "
            f"AOT={self.aot_550}"
        )


@dataclass
class OSOAADataset:
    """Container for OSOAA simulation results formatted for atmospheric correction.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths in nm.
    toa_reflectance : ndarray
        TOA reflectance (pi*Lt/F0) at each wavelength.
    true_rho_w : ndarray
        "True" water-leaving reflectance from OSOAA.
    geometry : GeometryAngles
        Observation geometry.
    ancillary : AncillaryData
        Ancillary data used in the simulation.
    scenario : SimulationScenario
        Full scenario parameters.
    radiance_components : dict, optional
        Per-wavelength radiance at key atmospheric levels from the Advanced
        output.  Keys are wavelength (int); values are dicts with keys
        ``toa``, ``above_surface``, ``below_surface``, etc.
    """

    wavelengths: np.ndarray
    toa_reflectance: np.ndarray
    true_rho_w: np.ndarray
    geometry: GeometryAngles
    ancillary: AncillaryData
    scenario: SimulationScenario
    radiance_components: Optional[Dict] = field(default=None, repr=False)

    def summary(self) -> None:
        """Print a human-readable summary."""
        print("OSOAA Simulation Dataset")
        print("=" * 50)
        print(f"Scenario: {self.scenario.name}")
        print(f"Wavelengths: {self.wavelengths} nm")
        print(f"\nGeometry:")
        print(f"  Solar Zenith: {self.geometry.solar_zenith}\u00b0")
        print(f"  View Zenith: {self.geometry.view_zenith}\u00b0")
        print(f"  Relative Azimuth: {self.geometry.relative_azimuth}\u00b0")
        print(f"\nOcean:")
        print(f"  Chlorophyll: {self.scenario.chlorophyll} mg/m\u00b3")
        print(f"\nAtmosphere:")
        print(f"  AOT (550 nm): {self.scenario.aot_550}")
        print(f"  Wind Speed: {self.ancillary.wind_speed} m/s")
        print(f"  Ozone: {self.ancillary.ozone} DU")
        if self.radiance_components:
            print(f"\nRadiance components: {len(self.radiance_components)} wavelengths")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def scale_aot_angstrom(
    aot_ref: float,
    wl_ref: float,
    wl_target: float,
    angstrom: float = 1.0,
) -> float:
    """Scale AOT using the Angstrom exponent law.

    AOT(lambda) = AOT(lambda_ref) * (lambda / lambda_ref) ** (-alpha)

    Parameters
    ----------
    aot_ref : float
        AOT at the reference wavelength.
    wl_ref : float
        Reference wavelength (nm).
    wl_target : float
        Target wavelength (nm).
    angstrom : float
        Angstrom exponent (typically 0.5--2.0 for maritime aerosols).

    Returns
    -------
    float
        AOT at the target wavelength.
    """
    return aot_ref * (wl_target / wl_ref) ** (-angstrom)


# ---------------------------------------------------------------------------
# OSOAA simulation wrapper
# ---------------------------------------------------------------------------

class OSOAASimulation:
    """Python wrapper for the OSOAA Fortran radiative transfer code.

    Supports TOA and subsurface radiance computations with automatic parsing
    of both standard (LUM_vsVZA) and advanced (LUM_Advanced_Up) output files.

    Note
    ----
    OSOAA is a scattering-only model.  It handles Rayleigh scattering,
    aerosols, the ocean surface, and hydrosols but does **not** include gas
    absorption (O3, NO2, H2O).  Gas transmittance corrections must be applied
    separately.

    Parameters
    ----------
    osoaa_root : Path or str
        Root directory of the OSOAA installation (contains ``exe/``).
    work_dir : Path or str, optional
        Working directory for simulation outputs.  A temporary directory is
        created if not specified.
    verbose : bool
        If True, print the working directory on construction.
    """

    def __init__(
        self,
        osoaa_root: Path | str,
        work_dir: Optional[Path | str] = None,
        verbose: bool = True,
    ):
        self.osoaa_root = Path(osoaa_root).resolve()
        self.exe_path = self.osoaa_root / "exe" / "OSOAA_MAIN.exe"

        if work_dir:
            self.work_dir = Path(work_dir).resolve()
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="osoaa_toa_"))

        # Sub-directories required by OSOAA
        self.mie_aer_dir = self.work_dir / "MIE_AER"
        self.mie_hyd_dir = self.work_dir / "MIE_HYD"
        self.surf_dir = self.work_dir / "SURF"
        for d in (self.mie_aer_dir, self.mie_hyd_dir, self.surf_dir):
            d.mkdir(parents=True, exist_ok=True)

        if not self.exe_path.exists():
            raise FileNotFoundError(
                f"OSOAA executable not found at {self.exe_path}"
            )

        if verbose:
            print(f"Working directory: {self.work_dir}")

    # -- parameter builders --------------------------------------------------

    def get_toa_params(
        self,
        wavelength_nm: float = 550.0,
        solar_zenith: float = 30.0,
        view_zenith: float = 0.0,
        relative_azimuth: float = 90.0,
        chlorophyll: float = 0.1,
        aot: float = 0.1,
        wind_speed: float = 5.0,
        pressure: float = 1013.25,
        extra_params: dict = None,
    ) -> Dict:
        """Return an OSOAA parameter dict for a TOA radiance simulation.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometres.
        solar_zenith, view_zenith, relative_azimuth : float
            Geometry angles in degrees.
        chlorophyll : float
            Chlorophyll-a concentration (mg m-3).
        aot : float
            Aerosol optical thickness at *this* wavelength.
        wind_speed : float
            Wind speed (m s-1) for the Cox--Munk surface.
        pressure : float
            Surface pressure (hPa).
        """
        wl_um = wavelength_nm / 1000.0

        params = {
            "OSOAA.ResRoot": str(self.work_dir),
            "OSOAA.Wa": wl_um,
            "ANG.Thetas": solar_zenith,
            "OSOAA.View.Phi": relative_azimuth,
            "OSOAA.View.Level": 1,  # TOA
            "OSOAA.View.Z": 0.0,
            # Atmosphere
            "AP.Pressure": pressure,
            "AP.HR": 8.0,
            "AP.HA": 2.0,
            # Aerosol (maritime mono-modal)
            "AER.DirMie": str(self.mie_aer_dir),
            "AER.Waref": wl_um,
            "AER.AOTref": aot,
            "AER.Model": 0,
            "AER.MMD.MRwa": 1.40,
            "AER.MMD.MIwa": -0.001,
            "AER.MMD.SDtype": 1,
            "AER.MMD.LNDradius": 0.10,
            "AER.MMD.LNDvar": 0.46,
            # Ocean
            "SEA.Depth": 1000.0,
            "HYD.DirMie": str(self.mie_hyd_dir),
            "HYD.Model": 1,
            "PHYTO.Chl": chlorophyll,
            "PHYTO.ProfilType": 1,
            "PHYTO.JD.slope": 4.0,
            "PHYTO.JD.rmin": 0.01,
            "PHYTO.JD.rmax": 200.0,
            "PHYTO.JD.MRwa": 1.05,
            "PHYTO.JD.MIwa": 0.0,
            "PHYTO.JD.rate": 1.0,
            "SED.Csed": 0.0,
            "YS.Abs440": 0.0,
            "DET.Abs440": 0.0,
            # Surface (Cox--Munk)
            "SEA.Dir": str(self.surf_dir),
            "SEA.Ind": 1.34,
            "SEA.Wind": wind_speed,
            "SEA.SurfAlb": 0.0,
            "SEA.BotType": 1,
            "SEA.BotAlb": 0.0,
            # Output files
            "OSOAA.ResFile.vsVZA": "LUM_vsVZA.txt",
            "OSOAA.ResFile.Adv.Up": "LUM_Advanced.txt",
        }
        
        # Extras
        if extra_params is not None:
            for key, item in extra_params.items():
                params[key] = item

                # Rayleigh?
                if key == 'AP.MOT' and item == 0.:
                    params.pop('AP.Pressure')
                    params.pop('AP.HR')
        
        # Return
        return params

    def get_subsurface_params(
        self,
        wavelength_nm: float = 550.0,
        solar_zenith: float = 30.0,
        chlorophyll: float = 0.1,
        wind_speed: float = 5.0,
    ) -> Dict:
        """Return an OSOAA parameter dict for a subsurface (0-) simulation.

        This gives the "true" water-leaving radiance that atmospheric
        correction should recover.
        """
        wl_um = wavelength_nm / 1000.0

        return {
            "OSOAA.ResRoot": str(self.work_dir),
            "OSOAA.Wa": wl_um,
            "ANG.Thetas": solar_zenith,
            "OSOAA.View.Phi": 90.0,
            "OSOAA.View.Level": 4,  # just below sea surface (0-)
            "OSOAA.View.Z": 0.0,
            "AP.Pressure": 1013.0,
            "AP.HR": 8.0,
            "AP.HA": 2.0,
            "AER.DirMie": str(self.mie_aer_dir),
            "AER.Waref": wl_um,
            "AER.AOTref": 0.01,
            "AER.Model": 0,
            "AER.MMD.MRwa": 1.40,
            "AER.MMD.MIwa": -0.001,
            "AER.MMD.SDtype": 1,
            "AER.MMD.LNDradius": 0.10,
            "AER.MMD.LNDvar": 0.46,
            "SEA.Depth": 1000.0,
            "HYD.DirMie": str(self.mie_hyd_dir),
            "HYD.Model": 1,
            "PHYTO.Chl": chlorophyll,
            "PHYTO.ProfilType": 1,
            "PHYTO.JD.slope": 4.0,
            "PHYTO.JD.rmin": 0.01,
            "PHYTO.JD.rmax": 200.0,
            "PHYTO.JD.MRwa": 1.05,
            "PHYTO.JD.MIwa": 0.0,
            "PHYTO.JD.rate": 1.0,
            "SED.Csed": 0.0,
            "YS.Abs440": 0.0,
            "DET.Abs440": 0.0,
            "SEA.Dir": str(self.surf_dir),
            "SEA.Ind": 1.34,
            "SEA.Wind": wind_speed,
            "SEA.SurfAlb": 0.0,
            "SEA.BotType": 1,
            "SEA.BotAlb": 0.0,
            "OSOAA.ResFile.vsVZA": "LUM_vsVZA_Lw.txt",
        }

    # -- execution -----------------------------------------------------------

    def build_command(self, params: Dict) -> List[str]:
        """Build the OSOAA command-line invocation."""
        cmd: List[str] = [str(self.exe_path)]
        for key, value in params.items():
            cmd.extend([f"-{key}", str(value)])
        return cmd

    def run(
        self,
        params: Dict,
        verbose: bool = True,
        timeout: int = 600,
    ) -> Dict:
        """Run OSOAA and return parsed results.

        Parameters
        ----------
        params : dict
            Parameter dict (from ``get_toa_params`` or ``get_subsurface_params``).
        verbose : bool
            Print progress messages.
        timeout : int
            Subprocess timeout in seconds.

        Returns
        -------
        dict
            Parsed results containing ``vza_data`` and optionally ``adv_up_data``.
        """
        cmd = self.build_command(params)

        if verbose:
            wl_nm = params.get("OSOAA.Wa", 0) * 1000
            level = params.get("OSOAA.View.Level", 1)
            level_name = {1: "TOA", 4: "Subsurface"}.get(
                level, f"Level {level}"
            )
            print(f"Running OSOAA ({level_name}) at {wl_nm:.0f} nm...")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.osoaa_root),
            env={**os.environ, "OSOAA_ROOT": str(self.osoaa_root)},
            timeout=timeout,
        )

        if result.returncode != 0:
            msg = f"OSOAA Error (return code {result.returncode}):"
            if result.stdout:
                msg += f"\nSTDOUT: {result.stdout[:2000]}"
            if result.stderr:
                msg += f"\nSTDERR: {result.stderr[:2000]}"
            raise RuntimeError(msg)

        if verbose:
            print("  Done.")

        return self.parse_results(params)

    # -- output parsing ------------------------------------------------------

    def parse_results(self, params: Dict) -> Dict:
        """Parse all OSOAA output files referenced in *params*."""
        results: Dict = {"params": params}

        # Standard VZA output
        std_dir = self.work_dir / "Standard_outputs"
        vza_filename = params.get("OSOAA.ResFile.vsVZA", "LUM_vsVZA.txt")
        vza_file = std_dir / vza_filename
        if vza_file.exists():
            results["vza_data"] = self._parse_vza_file(vza_file)
            results["output_file"] = str(vza_file)

        # Advanced Up output
        adv_filename = params.get("OSOAA.ResFile.Adv.Up")
        if adv_filename:
            adv_file = self.work_dir / "Advanced_outputs" / adv_filename
            if adv_file.exists():
                results["adv_up_data"] = self._parse_adv_file(adv_file)
                results["adv_up_file"] = str(adv_file)

        return results

    @staticmethod
    def _parse_vza_file(filepath: Path) -> Dict:
        """Parse a ``LUM_vsVZA*.txt`` file.

        Returns
        -------
        dict
            Arrays keyed by ``vza``, ``scattering_angle``, ``I``,
            ``reflectance``, ``DoLP``, ``I_pol``, ``refl_pol``.
        """
        with open(filepath, "r") as fh:
            lines = fh.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("VZA") and "SCA_ANG" in line:
                data_start = i + 1
                break

        data = np.loadtxt(filepath, skiprows=data_start)

        return {
            "vza": data[:, 0],
            "scattering_angle": data[:, 1],
            "I": data[:, 2],
            "reflectance": data[:, 3],
            "DoLP": data[:, 4],
            "I_pol": data[:, 5],
            "refl_pol": data[:, 6],
        }

    @staticmethod
    def _parse_adv_file(filepath: Path) -> Dict:
        """Parse an ``LUM_Advanced_Up*.txt`` (or Down) file.

        Columns: LEVEL  Z  VZA  SCA_ANG  I  Q  U  POL_ANG  POL_RATE  LPOL.

        Returns
        -------
        dict
            Arrays keyed by column name.
        """
        with open(filepath, "r") as fh:
            lines = fh.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if "LEVEL" in line and "VZA" in line and "SCA_ANG" in line:
                data_start = i + 1
                break

        data = np.loadtxt(filepath, skiprows=data_start)

        return {
            "level": data[:, 0].astype(int),
            "z": data[:, 1],
            "vza": data[:, 2],
            "scattering_angle": data[:, 3],
            "I": data[:, 4],
            "Q": data[:, 5],
            "U": data[:, 6],
            "pol_angle": data[:, 7],
            "pol_rate": data[:, 8],
            "lpol": data[:, 9],
        }

    # -- radiance extraction helpers -----------------------------------------

    @staticmethod
    def get_radiance_at_vza(results: Dict, target_vza: float) -> float:
        """Extract the Stokes-I radiance at a specific VZA from *results*."""
        if "vza_data" not in results:
            return np.nan
        vza = results["vza_data"]["vza"]
        I = results["vza_data"]["I"]
        idx = np.argmin(np.abs(vza - target_vza))
        return float(I[idx])

    @staticmethod
    def get_radiance_components_at_vza(
        adv_data: Dict,
        target_vza: float,
    ) -> Dict:
        """Extract upwelling radiance at key atmospheric levels.

        Returns the radiance at:

        - **TOA** (level 0, ~300 km)
        - **0+** just above the sea surface (last atmospheric level)
        - **0-** just below the sea surface (first ocean level)

        Parameters
        ----------
        adv_data : dict
            Parsed Advanced-Up data from ``_parse_adv_file``.
        target_vza : float
            Target viewing zenith angle (degrees).

        Returns
        -------
        dict
            Keys: ``toa``, ``above_surface``, ``below_surface``,
            ``z_toa``, ``z_above``, ``z_below``, ``vza_used``.
        """
        vza = adv_data["vza"]
        levels = adv_data["level"]
        z = adv_data["z"]
        I = adv_data["I"]

        positive_vza = vza[vza > 0]
        if len(positive_vza) == 0:
            positive_vza = np.abs(vza)
        unique_vza = np.unique(positive_vza)
        closest_vza = unique_vza[np.argmin(np.abs(unique_vza - target_vza))]

        mask = np.abs(vza - closest_vza) < 0.1
        f_levels = levels[mask]
        f_z = z[mask]
        f_I = I[mask]

        # TOA: level 0
        toa_idx = f_levels == 0
        toa_I = float(f_I[toa_idx][0]) if np.any(toa_idx) else np.nan
        toa_z = float(f_z[toa_idx][0]) if np.any(toa_idx) else np.nan

        # Above surface: smallest non-negative z
        atm_mask = f_z >= 0
        if np.any(atm_mask):
            above_idx = np.argmin(f_z[atm_mask])
            above_I = float(f_I[atm_mask][above_idx])
            above_z = float(f_z[atm_mask][above_idx])
        else:
            above_I = above_z = np.nan

        # Below surface: largest (least negative) z < 0
        ocean_mask = f_z < 0
        if np.any(ocean_mask):
            below_idx = np.argmax(f_z[ocean_mask])
            below_I = float(f_I[ocean_mask][below_idx])
            below_z = float(f_z[ocean_mask][below_idx])
        else:
            below_I = below_z = np.nan

        return {
            "toa": toa_I,
            "above_surface": above_I,
            "below_surface": below_I,
            "z_toa": toa_z,
            "z_above": above_z,
            "z_below": below_z,
            "vza_used": float(closest_vza),
        }

    @staticmethod
    def get_radiance_profile_at_vza(
        adv_data: Dict,
        target_vza: float,
    ) -> Dict:
        """Extract the full upwelling radiance vertical profile at a VZA.

        Returns
        -------
        dict
            Keys: ``level``, ``z``, ``I``, ``vza_used``.
        """
        vza = adv_data["vza"]
        unique_vza = np.unique(vza[vza > 0])
        closest_vza = unique_vza[np.argmin(np.abs(unique_vza - target_vza))]

        mask = np.abs(vza - closest_vza) < 0.1
        return {
            "level": adv_data["level"][mask],
            "z": adv_data["z"][mask],
            "I": adv_data["I"][mask],
            "vza_used": float(closest_vza),
        }

    # -- lifecycle -----------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the working directory."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Multi-wavelength simulation helpers
# ---------------------------------------------------------------------------

def run_toa_simulations(
    sim: OSOAASimulation,
    wavelengths: Sequence[int],
    scenario: SimulationScenario,
    angstrom: float = 1.0,
    verbose: bool = True,
    extra_params: dict = None,
) -> Dict[int, Dict]:
    """Run TOA simulations at multiple wavelengths for a given scenario.

    AOT is scaled spectrally from ``scenario.aot_550`` using the Angstrom law.

    Returns
    -------
    dict
        ``{wavelength_nm: results_dict}`` for each successful simulation.
    """
    toa_results: Dict[int, Dict] = {}
    for wl in wavelengths:
        aot_wl = scale_aot_angstrom(scenario.aot_550, 550.0, wl, angstrom)
        params = sim.get_toa_params(
            wavelength_nm=float(wl),
            solar_zenith=scenario.solar_zenith,
            view_zenith=scenario.view_zenith,
            relative_azimuth=scenario.relative_azimuth,
            chlorophyll=scenario.chlorophyll,
            aot=aot_wl,
            wind_speed=scenario.wind_speed,
            pressure=scenario.pressure,
            extra_params=extra_params,
        )
        params["OSOAA.ResFile.vsVZA"] = f"LUM_TOA_{wl}nm.txt"
        params["OSOAA.ResFile.Adv.Up"] = f"LUM_ADV_UP_TOA_{wl}nm.txt"
        try:
            toa_results[wl] = sim.run(params, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  ERROR at {wl} nm: {exc}")
        #embed(header='688 run_toa_simulations')
    return toa_results


def run_subsurface_simulations(
    sim: OSOAASimulation,
    wavelengths: Sequence[int],
    scenario: SimulationScenario,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """Run subsurface (water-leaving) simulations at multiple wavelengths.

    Returns
    -------
    dict
        ``{wavelength_nm: results_dict}`` for each successful simulation.
    """
    lw_results: Dict[int, Dict] = {}
    for wl in wavelengths:
        params = sim.get_subsurface_params(
            wavelength_nm=float(wl),
            solar_zenith=scenario.solar_zenith,
            chlorophyll=scenario.chlorophyll,
            wind_speed=scenario.wind_speed,
        )
        params["OSOAA.ResFile.vsVZA"] = f"LUM_Lw_{wl}nm.txt"
        try:
            lw_results[wl] = sim.run(params, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  ERROR at {wl} nm: {exc}")
    return lw_results


def extract_radiances(
    sim: OSOAASimulation,
    toa_results: Dict[int, Dict],
    target_vza: float,
) -> tuple[Dict[int, float], Dict[int, Dict]]:
    """Extract TOA radiances and Advanced-output components at a target VZA.

    Returns
    -------
    toa_radiances : dict
        ``{wavelength: float}`` TOA normalised radiance.
    radiance_components : dict
        ``{wavelength: dict}`` radiance at key levels (TOA, 0+, 0-).
    """
    toa_radiances: Dict[int, float] = {}
    radiance_components: Dict[int, Dict] = {}
    for wl in sorted(toa_results):
        res = toa_results[wl]
        toa_radiances[wl] = sim.get_radiance_at_vza(res, target_vza)
        if "adv_up_data" in res:
            radiance_components[wl] = sim.get_radiance_components_at_vza(
                res["adv_up_data"], target_vza
            )
    return toa_radiances, radiance_components


# ---------------------------------------------------------------------------
# Dataset creation and I/O
# ---------------------------------------------------------------------------

def create_dataset(
    toa_radiances: Dict[int, float],
    lw_radiances: Dict[int, float],
    scenario: SimulationScenario,
    radiance_components: Optional[Dict[int, Dict]] = None,
) -> OSOAADataset:
    """Package simulation results into an :class:`OSOAADataset`."""
    wavelengths = sorted(set(toa_radiances) & set(lw_radiances))
    toa_vals = np.array([toa_radiances[wl] for wl in wavelengths])
    lw_vals = np.array([lw_radiances[wl] for wl in wavelengths])

    geometry = GeometryAngles(
        solar_zenith=scenario.solar_zenith,
        solar_azimuth=0.0,
        view_zenith=scenario.view_zenith,
        view_azimuth=scenario.relative_azimuth,
    )
    ancillary = AncillaryData(
        pressure=scenario.pressure,
        wind_speed=scenario.wind_speed,
        ozone=scenario.ozone,
        water_vapor=1.5,
        relative_humidity=80.0,
    )

    return OSOAADataset(
        wavelengths=np.array(wavelengths),
        toa_reflectance=toa_vals,
        true_rho_w=lw_vals,
        geometry=geometry,
        ancillary=ancillary,
        scenario=scenario,
        radiance_components=radiance_components,
    )


def save_dataset(
    dataset: OSOAADataset,
    filepath: Path | str,
    radiance_components: Optional[Dict] = None,
) -> None:
    """Save an :class:`OSOAADataset` to a JSON file.

    Parameters
    ----------
    dataset : OSOAADataset
        The dataset to serialise.
    filepath : Path or str
        Destination JSON path.
    radiance_components : dict, optional
        Override for radiance components (if not already on *dataset*).
    """
    components = radiance_components or dataset.radiance_components

    data = {
        "wavelengths_nm": dataset.wavelengths.tolist(),
        "toa_reflectance": dataset.toa_reflectance.tolist(),
        "true_rho_w": dataset.true_rho_w.tolist(),
        "geometry": {
            "solar_zenith": float(dataset.geometry.solar_zenith),
            "solar_azimuth": float(dataset.geometry.solar_azimuth),
            "view_zenith": float(dataset.geometry.view_zenith),
            "view_azimuth": float(dataset.geometry.view_azimuth),
        },
        "ancillary": {
            "pressure": float(dataset.ancillary.pressure),
            "wind_speed": float(dataset.ancillary.wind_speed),
            "ozone": float(dataset.ancillary.ozone),
            "water_vapor": float(dataset.ancillary.water_vapor),
            "relative_humidity": float(dataset.ancillary.relative_humidity),
        },
        "scenario": {
            "name": dataset.scenario.name,
            "chlorophyll": dataset.scenario.chlorophyll,
            "aot_550": dataset.scenario.aot_550,
        },
    }

    if components:
        comp_out: Dict = {}
        for wl in sorted(components):
            c = components[wl]
            comp_out[str(wl)] = {
                "toa": c["toa"],
                "above_surface": c["above_surface"],
                "below_surface": c["below_surface"],
                "z_toa_m": c["z_toa"],
                "z_above_m": c["z_above"],
                "z_below_m": c["z_below"],
                "vza_used": c["vza_used"],
            }
        data["radiance_components"] = comp_out

    with open(filepath, "w") as fh:
        json.dump(data, fh, indent=2)


def load_dataset(filepath: Path | str) -> Dict:
    """Load an OSOAA dataset from a JSON file.

    Returns
    -------
    dict
        With numpy arrays for ``wavelengths_nm``, ``toa_reflectance``,
        ``true_rho_w``, plus ``geometry``, ``ancillary``, ``scenario``,
        and optionally ``radiance_components``.
    """
    with open(filepath, "r") as fh:
        data = json.load(fh)

    data["wavelengths_nm"] = np.array(data["wavelengths_nm"])
    data["toa_reflectance"] = np.array(data["toa_reflectance"])
    data["true_rho_w"] = np.array(data["true_rho_w"])
    return data
