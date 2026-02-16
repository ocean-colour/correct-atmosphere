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
from correct_atmosphere.osoaa import OSOAASimulation
from correct_atmosphere.osoaa import SimulationScenario
from correct_atmosphere.osoaa import scale_aot_angstrom
from correct_atmosphere.osoaa import run_subsurface_simulations
from correct_atmosphere.osoaa import run_toa_simulations
from correct_atmosphere.osoaa import extract_radiances
from correct_atmosphere.osoaa import create_dataset
from correct_atmosphere.osoaa import save_dataset


# MODIS-Aqua ocean colour bands (nm)
MODIS_WAVELENGTHS_NM: List[int] = [
    412, 443, 469, 488, 531, 547, 667, 678, 748, 869
]

def init_sim():
    # Configure OSOAA paths
    # Adjust this path to your OSOAA installation
    OSOAA_ROOT = Path(os.environ.get(
        'OSOAA_ROOT', 
        '/home/xavier/Oceanography/RadiativeTransferCode-OSOAA'
    )).resolve()

    OSOAA_EXE = OSOAA_ROOT / 'exe' / 'OSOAA_MAIN.exe'

    # Initialize simulation
    sim = OSOAASimulation(OSOAA_ROOT)

    print(f"OSOAASimulation ready")
    print(f"  Executable: {sim.exe_path}")
    print(f"  Working dir: {sim.work_dir}")

    return sim

def define_scenario(name="Open Ocean - Moderate", solar_zenith=30.0,
        view_zenith=15.0, relative_azimuth=90.0,
        chlorophyll=0.3, aot_550=0.10,
        wind_speed=5.0, pressure=1013.25, ozone=300.0):

    # Define our test scenario
    scenario = SimulationScenario(
        name=name,
        solar_zenith=solar_zenith,
        view_zenith=view_zenith,
        relative_azimuth=relative_azimuth,
        chlorophyll=chlorophyll,
        aot_550=aot_550,
        wind_speed=wind_speed,
        pressure=pressure,
        ozone=ozone,
    )

    print("Simulation scenario:")
    print(scenario)

    return scenario


def main(outfile:str, no_rayleigh:bool=False):
    # Init
    sim = init_sim()
    scenario = define_scenario()

    # Modify sim
    if no_rayleigh:
        extra_toa = {'AP.MOT': 0.}
    else:
        extra_toa = None

    # Use MODIS-Aqua ocean colour bands from the module
    # MODIS-Aqua ocean colour bands (nm)
    MODIS_WAVELENGTHS_NM: List[int] = [
        412, 443, 469, 488, 531, 547, 667, 678, 748, 869
    ]
    WAVELENGTHS_NM = MODIS_WAVELENGTHS_NM

    # Demonstrate AOT scaling with Angstrom exponent
    angstrom_exp = 1.0  # Typical for maritime aerosol
    print("AOT at each wavelength (Angstrom exponent = 1.0):")
    print("-" * 40)
    for wl in WAVELENGTHS_NM:
        aot = scale_aot_angstrom(scenario.aot_550, 550.0, wl, angstrom_exp)
        print(f"  {wl:3d} nm: AOT = {aot:.4f}")
    
    # Run TOA simulations at all wavelengths
    print("Running TOA radiance simulations...")
    print("=" * 50)

    toa_results = run_toa_simulations(
        sim, WAVELENGTHS_NM, scenario, angstrom=angstrom_exp,
        extra_params=extra_toa
    )

    print(f"\nTOA simulations completed!")
    print(f"Successful: {len(toa_results)} / {len(WAVELENGTHS_NM)} wavelengths")

    # Extract TOA radiances and radiance components at the satellite viewing angle
    target_vza = scenario.view_zenith

    toa_radiances, radiance_components = extract_radiances(
        sim, toa_results, target_vza)

    print(f"TOA Radiances at VZA = {target_vza}:")
    print("=" * 50)
    print(f"{'Wavelength (nm)':<18} {'TOA Radiance':>15}")
    print("-" * 50)
    for wl in sorted(toa_radiances.keys()):
        rho_toa = toa_radiances[wl]
        print(f"{wl:>8} nm        {rho_toa:>15.6f}")
    print("-" * 50)

    # Run water-leaving (subsurface) simulations
    print("Running water-leaving radiance simulations...")
    print("=" * 50)

    lw_results = run_subsurface_simulations(sim, WAVELENGTHS_NM, scenario)

    print("\nWater-leaving simulations completed!")

    # Extract water-leaving radiances at nadir
    lw_radiances = {}

    print(f"Water-Leaving Radiances (subsurface at nadir):")
    print("=" * 50)
    print(f"{'Wavelength (nm)':<18} {'Lw (normalized)':>18}")
    print("-" * 50)

    for wl in sorted(lw_results.keys()):
        rho_lw = sim.get_radiance_at_vza(lw_results[wl], 0.0)
        lw_radiances[wl] = rho_lw
        print(f"{wl:>8} nm        {rho_lw:>18.6f}")

    print("-" * 50)

    # Create dataset (OSOAADataset, create_dataset imported from correct_atmosphere.osoaa)
    if toa_radiances and lw_radiances:
        dataset = create_dataset(
            toa_radiances, lw_radiances, scenario,
            radiance_components=radiance_components,
        )
        dataset.summary()

    # Save the dataset (save_dataset / load_dataset imported from correct_atmosphere.osoaa)
    if 'dataset' in dir():
        output_path = Path('.') / outfile
        save_dataset(dataset, output_path)
        print(f"Dataset saved to: {output_path}")

# Command line entry point
if __name__ == "__main__":
    
    # Standard scenario
    #main(outfile="osoaa_toa_dataset_std.json")

    # No Rayleigh scenario
    main(outfile="osoaa_toa_dataset_no_rayleigh.json",
        no_rayleigh=True)