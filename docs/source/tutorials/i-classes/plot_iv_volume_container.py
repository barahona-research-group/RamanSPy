"""
Storing volumetric data
--------------------------------------

The management of volumetric Raman spectroscopic data in `RamanSPy` is guided through the :class:`ramanspy.SpectralVolume` class.

.. seealso:: As the :class:`ramanspy.SpectralVolume` class extends the :class:`ramanspy.SpectralContainer` class, most of its
             functionality is inherited from this class. Hence, users are advised to first check the documentation of
             the :class:`ramanspy.Spectrum` class and the :ref:`Generic data container` tutorial.

Below, we will inspect some of the main features the :class:`ramanspy.SpectralVolume` class provides on top of those inherited
through the :class:`ramanspy.SpectralContainer` class.
"""
import numpy as np
import ramanspy


# %%
# We can define a spectrum by providing a relevant 4D intensity data array and the corresponding Raman wavenumber
# axis, just as we initialise :class:`ramanspy.SpectralContainer` instances. As an example, we will create a 50x50x10 spectroscopic
# volume, each point of which contains a Raman spectrum containing 1500 spectral points.
#
spectral_data = np.random.rand(50, 50, 10, 1500)
spectral_axis = np.linspace(100, 3600, 1500)

raman_volume = ramanspy.SpectralVolume(spectral_data, spectral_axis)

# %%
# Then, we can use all features of the :class:`ramanspy.SpectralContainer` class as usual. For instance,
raman_volume.shape

# %%
# Another way to create :class:`ramanspy.SpectralVolume` instances is by stacking :class:`ramanspy.SpectralImages` instances along
# the z-axis. When doing that, the spatial dimensions of the images and their spectral axes must match.
raman_images = [ramanspy.SpectralImage(np.random.rand(50, 50, 1500), spectral_axis) for _ in range(5)]
raman_volume = ramanspy.SpectralVolume.from_image_stack(raman_images)

# %%
raman_volume.shape

# %%
# Once we have initialised a :class:`ramanspy.SpectralVolume` instance, we can visualise spectral slices across it
raman_volume.plot(bands=[1500, 2500])


# %%
# as well as access individual layers from it and plot them
raman_image = raman_volume.layer(3)

raman_image.plot(bands=1500)
