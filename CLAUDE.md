# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

correct-atmosphere is a Python package for performing atmospheric corrections for ocean color remote sensing. This is an early-stage project with the package structure initialized but core functionality yet to be implemented.

## Development Commands

```bash
# Install in development mode
pip install -e .

# Run tests
python setup.py pytest
# Or directly with pytest
pytest

# Run a single test
pytest tests/test_module.py::test_function -v
```

## Dependencies

The package requires Python >=3.11 and includes scientific computing libraries:
- xarray, h5netcdf for data handling
- scikit-learn, scikit-image for ML/image processing
- healpy for spherical projections
- pysolar for solar position calculations
- emcee, corner for MCMC sampling
- boto3, smart-open[s3] for S3 data access
- timm for deep learning models

## Architecture

The main package is `correct_atmosphere/`. As the project develops, atmospheric correction algorithms and utilities will be added here.
