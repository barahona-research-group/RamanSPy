"""
Storing spectra
--------------------------------------

The management of single Raman spectra in `RamanSPy` is guided through the :class:`ramanspy.Spectrum` class.

.. seealso:: As the :class:`ramanspy.Spectrum` class extends the :class:`ramanspy.SpectralContainer` class, most of its
             functionality is inherited from this class. Hence, users are advised to first check the documentation of
             the :class:`ramanspy.Spectrum` class and the :ref:`Generic data container` tutorial.

Below, we will inspect some of the main features the :class:`ramanspy.Spectrum` class provides on top of those inherited
through the :class:`ramanspy.SpectralContainer` class.
"""

import numpy as np
import ramanspy

# %%
# We can define a spectrum by providing a relevant 1D intensity data array and the corresponding ramanspy wavenumber
# axis, just as we initialise :class:`raman.SpectralContainer` instances. As an example, we will create a Raman spectrum containing 1500 spectral points.
spectral_axis = np.linspace(100, 3600, 1500)
spectral_data = np.sin(spectral_axis/120)

raman_spectrum = ramanspy.Spectrum(spectral_data, spectral_axis)

# %%
# Then, we can use all features of the :class:`ramanspy.SpectralContainer` class as usual. For instance,
raman_spectrum.shape

# %%
# At the moment, the only functionality the :class:`ramanspy.Spectrum` class provides over :class:`ramanspy.SpectralContainer` is
# the highly-customisable :meth:`ramanspy.Spectrum.plot`, which can be used to quickly visualise spectral slices across
# Raman spectroscopic images.
raman_spectrum.plot()

