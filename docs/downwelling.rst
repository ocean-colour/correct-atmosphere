.. _downwelling:

======================
Downwelling Irradiance
======================

This section describes the downwelling irradiance calculations implemented in
the ``correct_atmosphere.downwelling`` module, which provides Ed(0+) values
essential for computing remote-sensing reflectance.

Overview
========

The downwelling irradiance at the sea surface, Ed(0+), is required to convert
water-leaving radiance to remote-sensing reflectance:

.. math::

    R_{rs} = \frac{L_w}{E_d(0^+)}

The downwelling irradiance consists of two components:

.. math::

    E_d(0^+) = E_d^{dir} + E_d^{dif}

where :math:`E_d^{dir}` is the direct beam component and :math:`E_d^{dif}` is
the diffuse (sky) component.

Solar Irradiance Data
=====================

TSIS-1 Hybrid Solar Reference Spectrum
--------------------------------------

The module uses the TSIS-1 Hybrid Solar Reference Spectrum (HSRS) v2 as the
default source for extraterrestrial solar irradiance F0. This is the current
NASA standard for ocean color processing.

**Key characteristics:**

* Wavelength range: 202-2730 nm
* Spectral resolution: 0.1 nm (25,281 data points)
* Includes measurement uncertainty
* Based on TSIS-1 Spectral Irradiance Monitor (SIM) measurements

**Reference:** Coddington, O., et al. (2021). The TSIS-1 Hybrid Solar Reference
Spectrum. Geophysical Research Letters, 48, e2020GL091709.
https://doi.org/10.1029/2020GL091709

Data File Requirement
---------------------

The TSIS-1 HSRS data must be present at::

    correct_atmosphere/data/Solar/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc

Download from: https://lasp.colorado.edu/lisird/data/tsis1_hsrs

Typical F0 Values
-----------------

Extraterrestrial solar irradiance at key ocean color wavelengths
(in mW cm\ :sup:`-2` μm\ :sup:`-1`):

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Wavelength (nm)
     - F0
     - Application
   * - 412
     - ~187
     - Chlorophyll absorption, CDOM
   * - 443
     - ~197
     - Chlorophyll absorption peak
   * - 490
     - ~208
     - Chlorophyll, carotenoids
   * - 555
     - ~193
     - Minimum absorption
   * - 670
     - ~153
     - Chlorophyll fluorescence
   * - 865
     - ~96
     - NIR aerosol correction

Unit Conversion
---------------

The native TSIS-1 HSRS units are W m\ :sup:`-2` nm\ :sup:`-1`. The module
converts to the ocean color standard of mW cm\ :sup:`-2` μm\ :sup:`-1`:

.. math::

    1 \text{ W m}^{-2} \text{ nm}^{-1} = 100 \text{ mW cm}^{-2} \mu\text{m}^{-1}

Derivation:

* Power: 1 W = 10\ :sup:`3` mW
* Area: 1 m\ :sup:`-2` = 10\ :sup:`-4` cm\ :sup:`-2`
* Wavelength: 1 nm\ :sup:`-1` = 10\ :sup:`3` μm\ :sup:`-1`
* Combined: 10\ :sup:`3` × 10\ :sup:`-4` × 10\ :sup:`3` = 100

Ed Calculation
==============

Direct Beam Component
---------------------

The direct beam irradiance reaching the sea surface is:

.. math::

    E_d^{dir} = F_0 \cdot f_d \cdot \cos\theta_s \cdot T_{dir}

where:

* :math:`F_0` - Extraterrestrial solar irradiance
* :math:`f_d` - Earth-Sun distance correction factor
* :math:`\theta_s` - Solar zenith angle
* :math:`T_{dir}` - Direct beam transmittance

The direct transmittance accounts for:

* Rayleigh scattering
* Aerosol extinction
* Gas absorption (O₃, NO₂, H₂O)

.. math::

    T_{dir} = \exp\left[-(\tau_R + \tau_A + \tau_{gas}) / \cos\theta_s\right]

Diffuse Component
-----------------

The diffuse (sky) irradiance is computed from:

.. math::

    E_d^{dif} = F_0 \cdot f_d \cdot \cos\theta_s \cdot T_{dif}

where :math:`T_{dif}` is the diffuse transmittance that accounts for
multiple scattering. The diffuse component is larger at shorter wavelengths
due to stronger Rayleigh scattering.

Earth-Sun Distance Correction
-----------------------------

The Earth-Sun distance varies throughout the year, affecting the solar
irradiance by the inverse square law:

.. math::

    f_d = \left(\frac{R_0}{R}\right)^2

where :math:`R_0` = 1 AU (mean Earth-Sun distance) and :math:`R` is the
actual distance. This correction varies by approximately ±3.4% over the year:

* Perihelion (~Jan 3): :math:`f_d` ≈ 1.034
* Aphelion (~Jul 4): :math:`f_d` ≈ 0.967

PACE Support
============

The module provides native support for NASA PACE (Plankton, Aerosol, Cloud,
ocean Ecosystem) mission hyperspectral data:

**PACE OCI Hyperspectral:**

* Wavelength range: 340-890 nm
* Resolution: 5 nm
* Bands: 111 hyperspectral channels

**PACE OCI SWIR:**

* Discrete bands at: 940, 1038, 1250, 1378, 1615, 2130, 2260 nm

**Implementation:**

.. code-block:: python

    from correct_atmosphere.downwelling import (
        get_pace_wavelengths,
        downwelling_irradiance_spectral,
        PACE_OCI_WAVELENGTHS,
    )

    # Get PACE wavelengths
    wavelengths = get_pace_wavelengths(include_swir=False)

    # Calculate spectral Ed for PACE
    result = downwelling_irradiance_spectral(
        solar_zenith=30.0,
        wavelength_range=(340, 890),
        resolution=5.0,
        aerosol_tau_550=0.1,
    )

    ed = result["ed"]
    ed_direct = result["ed_direct"]
    ed_diffuse = result["ed_diffuse"]

Usage Examples
==============

Basic Ed Calculation
--------------------

.. code-block:: python

    from correct_atmosphere.downwelling import (
        downwelling_irradiance,
        extraterrestrial_solar_irradiance,
    )

    # Single wavelength
    ed = downwelling_irradiance(
        wavelength=550.0,
        solar_zenith=30.0,
        aerosol_tau=0.1,
        day_of_year=172,
    )
    print(f"Ed at 550 nm: {ed:.2f} mW cm^-2 um^-1")

    # Multiple wavelengths
    import numpy as np
    wavelengths = np.array([443, 490, 555, 670, 865])
    ed = downwelling_irradiance(wavelengths, solar_zenith=30.0)

Working with the Solar Spectrum
-------------------------------

.. code-block:: python

    from correct_atmosphere.downwelling import get_solar_spectrum

    # Load full spectrum
    spectrum = get_solar_spectrum()
    print(f"Range: {spectrum.wavelengths[0]}-{spectrum.wavelengths[-1]} nm")
    print(f"Points: {len(spectrum.wavelengths)}")

    # Load subset for visible range
    vis_spectrum = get_solar_spectrum(wavelength_range=(400, 700))

    # Interpolate to specific wavelengths
    f0_443 = spectrum.interpolate(443.0)
    print(f"F0 at 443 nm: {f0_443:.2f} mW cm^-2 um^-1")

    # Access uncertainty (TSIS-1 HSRS only)
    if spectrum.uncertainty is not None:
        idx = np.argmin(np.abs(spectrum.wavelengths - 443))
        unc = spectrum.uncertainty[idx]
        print(f"Uncertainty: ±{unc:.2f} mW cm^-2 um^-1")

    # Convert to SI units
    spectrum_si = spectrum.to_si_units()
    f0_si = spectrum_si.interpolate(443.0)
    print(f"F0 at 443 nm: {f0_si:.4f} W m^-2 nm^-1")

Computing Remote-Sensing Reflectance
------------------------------------

.. code-block:: python

    from correct_atmosphere.downwelling import downwelling_irradiance

    # Given water-leaving radiance Lw (e.g., from atmospheric correction)
    Lw = 0.5  # mW cm^-2 um^-1 sr^-1

    # Compute Ed at the same wavelength
    Ed = downwelling_irradiance(
        wavelength=555.0,
        solar_zenith=30.0,
        aerosol_tau=0.05,
    )

    # Remote-sensing reflectance
    Rrs = Lw / Ed
    print(f"Rrs(555) = {Rrs:.6f} sr^-1")

Spectral Ed with Components
---------------------------

.. code-block:: python

    from correct_atmosphere.downwelling import downwelling_irradiance_spectral
    import matplotlib.pyplot as plt

    # Compute spectral Ed
    result = downwelling_irradiance_spectral(
        solar_zenith=45.0,
        wavelength_range=(400, 800),
        resolution=5.0,
        aerosol_tau_550=0.15,
    )

    # Plot components
    plt.figure(figsize=(10, 6))
    plt.plot(result["wavelengths"], result["ed"], label="Total Ed")
    plt.plot(result["wavelengths"], result["ed_direct"], label="Direct")
    plt.plot(result["wavelengths"], result["ed_diffuse"], label="Diffuse")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Ed (mW cm$^{-2}$ μm$^{-1}$)")
    plt.legend()
    plt.title("Downwelling Irradiance Components")

API Reference
=============

Classes
-------

.. autoclass:: correct_atmosphere.downwelling.SolarSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: correct_atmosphere.downwelling.get_solar_spectrum

.. autofunction:: correct_atmosphere.downwelling.extraterrestrial_solar_irradiance

.. autofunction:: correct_atmosphere.downwelling.solar_zenith_factor

.. autofunction:: correct_atmosphere.downwelling.downwelling_irradiance

.. autofunction:: correct_atmosphere.downwelling.downwelling_irradiance_direct

.. autofunction:: correct_atmosphere.downwelling.downwelling_irradiance_diffuse

.. autofunction:: correct_atmosphere.downwelling.downwelling_irradiance_spectral

.. autofunction:: correct_atmosphere.downwelling.get_pace_wavelengths

.. autofunction:: correct_atmosphere.downwelling.convert_irradiance_units

.. autofunction:: correct_atmosphere.downwelling.clear_solar_spectrum_cache

Constants
---------

.. py:data:: correct_atmosphere.downwelling.PACE_OCI_WAVELENGTHS

   Array of PACE OCI hyperspectral wavelengths (340-890 nm at 5 nm resolution).

.. py:data:: correct_atmosphere.downwelling.PACE_OCI_SWIR_BANDS

   Array of PACE OCI SWIR band center wavelengths.

References
==========

1. Mobley, C.D., et al. (2016). Ocean Optics Web Book. NASA/TM-2016-217551.

2. Coddington, O., et al. (2021). The TSIS-1 Hybrid Solar Reference Spectrum.
   Geophysical Research Letters, 48, e2020GL091709.
   https://doi.org/10.1029/2020GL091709

3. Thuillier, G., et al. (2003). The solar spectral irradiance from 380 to
   2500 nm as measured by the SOLSPEC spectrometer from the ATLAS and EURECA
   missions. Solar Physics, 214:1-22.

4. NASA PACE Mission: https://pace.gsfc.nasa.gov/
