# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

correct-atmosphere is a Python package for performing atmospheric corrections for ocean color remote sensing. The package implements algorithms based on NASA's ocean color processing methodology (reference: NASA TM-2016-217551 in `docs/`).

## Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=correct_atmosphere --cov-report=term-missing

# Run a single test file
pytest correct_atmosphere/tests/test_rayleigh.py -v

# Run a specific test
pytest correct_atmosphere/tests/test_rayleigh.py::TestRayleighOpticalThickness -v

# Linting and formatting
black correct_atmosphere/
ruff check correct_atmosphere/

# Type checking
mypy correct_atmosphere/ --ignore-missing-imports

# Build documentation
cd docs && make html
```

## Dependencies

The package requires Python >=3.9 (CI tests 3.11, 3.12, 3.13). Core dependencies:
- numpy, scipy for numerical computing
- xarray, netCDF4 for data handling

Dev dependencies (install with `pip install -e ".[dev]"`):
- pytest, pytest-cov for testing
- black, ruff for formatting/linting
- mypy for type checking

Docs dependencies (install with `pip install -e ".[docs]"`):
- sphinx, sphinx-rtd-theme, numpydoc

## Architecture

The package is in `correct_atmosphere/` with the following modules:

| Module | Purpose |
|--------|---------|
| `correction.py` | Main `AtmosphericCorrection` class orchestrating the full correction pipeline |
| `rayleigh.py` | Rayleigh scattering and optical thickness calculations |
| `aerosols.py` | Aerosol models and lookup tables |
| `gases.py` | Gas absorption (ozone, NO2) transmittance |
| `glint.py` | Sun glint estimation using Cox-Munk model |
| `whitecaps.py` | Whitecap fraction and reflectance |
| `transmittance.py` | Atmospheric transmittance (direct/diffuse) |
| `normalization.py` | BRDF correction and reflectance normalization |
| `polarization.py` | Polarization correction calculations |
| `outofband.py` | Out-of-band correction for sensor response |
| `constants.py` | Physical constants and sensor band definitions (SeaWiFS, MODIS, VIIRS) |

Tests are in `correct_atmosphere/tests/` with one test file per module.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- Tests on Ubuntu/macOS/Windows with Python 3.11-3.13
- NumPy dev compatibility testing
- Linting (black, ruff)
- Type checking (mypy)
- Documentation build

## Documentation

- Sphinx docs in `docs/` (hosted on ReadTheDocs)
- Tutorial notebooks in `nb/`:
  - `01_getting_started.ipynb` - Basic usage
  - `02_physical_components.ipynb` - Physical components
  - `03_aerosols_transmittance.ipynb` - Aerosol/transmittance
  - `04_full_correction.ipynb` - Complete workflow
