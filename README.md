# correct-atmosphere

[![Documentation Status](https://readthedocs.org/projects/correct-atmosphere/badge/?version=latest)](https://correct-atmosphere.readthedocs.io/en/latest/?badge=latest)

Python package for performing atmospheric corrections for ocean color remote sensing.

This package implements the NASA Ocean Biology Processing Group (OBPG) atmospheric correction algorithms documented in:

> Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
> *Atmospheric Correction for Satellite Ocean Color Radiometry*.
> NASA Technical Memorandum 2016-217551.

A copy of the NASA technical memorandum is included in `docs/NASA-TM-2016-217551.pdf`.

## Documentation

Full documentation is available at **[correct-atmosphere.readthedocs.io](https://correct-atmosphere.readthedocs.io/)**

The documentation includes:

- [Installation guide](https://correct-atmosphere.readthedocs.io/en/latest/installation.html)
- [Quick start guide](https://correct-atmosphere.readthedocs.io/en/latest/quickstart.html)
- [Tutorials with Jupyter notebooks](https://correct-atmosphere.readthedocs.io/en/latest/tutorials.html)
- [Theory and background](https://correct-atmosphere.readthedocs.io/en/latest/theory.html)
- [API reference](https://correct-atmosphere.readthedocs.io/en/latest/api.html)

## Installation

```bash
pip install correct_atmosphere
```

For development:

```bash
git clone https://github.com/ocean-colour/correct-atmosphere.git
cd correct-atmosphere
pip install -e ".[dev]"
```

## Features

- **Complete atmospheric correction pipeline**: From TOA radiance to water-leaving reflectance
- **Multi-sensor support**: SeaWiFS, MODIS-Aqua, MODIS-Terra, VIIRS-NPP, VIIRS-NOAA20
- **Modular design**: Use individual correction components or the full pipeline
- **Well-documented**: Extensive docstrings with equation references to NASA TM-2016-217551

## Quick Example

```python
from correct_atmosphere.rayleigh import rayleigh_optical_thickness
from correct_atmosphere.gases import ozone_transmittance

# Rayleigh optical thickness at 443 nm
tau_r = rayleigh_optical_thickness(443.0)

# Ozone transmittance
t_o3 = ozone_transmittance(443.0, 350.0, 30.0, 15.0)
```

## Requirements

- Python >= 3.9
- NumPy, SciPy
- xarray, netCDF4

## License

See [LICENSE](LICENSE) for details.
