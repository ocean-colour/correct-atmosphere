.. _tutorials:

=========
Tutorials
=========

Interactive Jupyter notebooks are provided in the ``nb/`` directory to help you
learn the atmospheric correction algorithms step-by-step. These tutorials cover
everything from basic concepts to the full correction workflow.

Notebook Overview
-----------------

The tutorials are designed to be worked through in order:

1. **Getting Started** (``01_getting_started.ipynb``)

   Introduction to atmospheric correction for ocean color remote sensing.

   - Overview of the atmospheric correction problem
   - Understanding TOA radiance components
   - Introduction to the correction workflow
   - Basic usage of the ``correct_atmosphere`` package

2. **Physical Components** (``02_physical_components.ipynb``)

   Deep dive into the physical components of atmospheric correction.

   - Rayleigh scattering calculations and optical thickness
   - Gas absorption (O3, NO2) and transmittance
   - Sun glint estimation using the Cox-Munk model
   - Whitecap/foam reflectance contributions

3. **Aerosols and Transmittance** (``03_aerosols_transmittance.ipynb``)

   Aerosol correction and atmospheric transmittance.

   - Aerosol optical properties and models
   - The Angstrom exponent and spectral dependence
   - Black-pixel algorithm for clear oceanic waters
   - Non-black-pixel correction for turbid waters
   - Direct and diffuse atmospheric transmittance

4. **Full Atmospheric Correction** (``04_full_correction.ipynb``)

   Complete end-to-end atmospheric correction workflow.

   - Setting up the ``AtmosphericCorrection`` processor
   - Preparing input data (TOA radiances, geometry, ancillary)
   - Running the full correction pipeline
   - Understanding outputs (Rrs, nLw, chlorophyll)
   - Quality flags and diagnostics
   - Processing imagery (2D arrays)

Running the Tutorials
---------------------

To run the tutorials, navigate to the ``nb/`` directory and start Jupyter:

.. code-block:: bash

    cd nb/
    jupyter notebook

Or using JupyterLab:

.. code-block:: bash

    cd nb/
    jupyter lab

Prerequisites
-------------

The tutorials require the package to be installed with its dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Additionally, ensure you have Jupyter installed:

.. code-block:: bash

    pip install jupyter

Learning Path
-------------

**For beginners**: Start with Notebook 1 and work through sequentially. Each
notebook builds on concepts from the previous ones.

**For experienced users**: Jump directly to Notebook 4 for the full correction
workflow, referring back to earlier notebooks for details on specific components.

**For algorithm developers**: Notebooks 2 and 3 provide detailed coverage of
individual correction components with references to the NASA TM-2016-217551
documentation.

References
----------

The tutorials follow the algorithms documented in:

    Mobley, C.D., Werdell, J., Franz, B., Ahmad, Z., and Bailey, S. (2016).
    *Atmospheric Correction for Satellite Ocean Color Radiometry*.
    NASA Technical Memorandum 2016-217551.

A copy of this document is available in ``docs/NASA-TM-2016-217551.pdf``.
