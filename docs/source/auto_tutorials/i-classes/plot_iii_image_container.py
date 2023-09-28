"""
Storing imaging data
--------------------------------------

The management of imaging Raman spectroscopic data in `RamanSPy` is guided through the :class:`ramanspy.SpectralImage` class.

.. seealso:: As the :class:`ramanspy.SpectralImage` class extends the :class:`ramanspy.SpectralContainer` class, most of its
             functionality is inherited from this class. Hence, users are advised to first check the documentation of
             the :class:`ramanspy.SpectralImage` class and the :ref:`Generic data container` tutorial.

Below, we will inspect some of the main features the :class:`ramanspy.SpectralImage` class provides on top of those inherited
through the :class:`ramanspy.SpectralContainer` class.
"""
import numpy as np
import ramanspy


# %%
# We can define a spectral image by providing the relevant 3D intensity data array and the corresponding Raman wavenumber
# axis, just as we initialise :class:`ramanspy.SpectralContainer` instances. As an example, we will create a 50x50 spectroscopic
# image, each point of which contains a Raman spectrum containing 1500 spectral points.
spectral_data = np.random.rand(50, 50, 1500)
spectral_axis = np.linspace(100, 3600, 1500)

raman_image = ramanspy.SpectralImage(spectral_data, spectral_axis)

# %%
# Then, we can use all features of the :class:`ramanspy.SpectralContainer` class as usual. For instance,
raman_image.shape

# %%
# At the moment, the only functionality the :class:`ramanspy.SpectralImage` class provides over :class:`ramanspy.SpectralContainer`
# is the highly-customisable :meth:`ramanspy.SpectralImage.plot`, which can be used to quickly visualise spectral slices
# across Raman spectroscopic images.

# spectral slices across 1500 cm^-1 and 2500 cm^-1
raman_image.plot(bands=[1500, 2500])

# %%
# We can also plot individual spectra within the image using indexing and :class:`ramanspy.Spectrum`'s plot method.
raman_image[25, 25].plot()
