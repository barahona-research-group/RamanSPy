"""
Storing generic data
--------------------------------------

The backbone of `RamanSPy's` data management core is the :class:`ramanspy.SpectralContainer` class. It serves as a generic data container
which defines the main Raman spectroscopic data-related features and functionalities.

The data stored within a :class:`ramanspy.SpectralContainer` instance can be of any dimension, but if you are dealing with single
spectra, imaging and volumetric Raman data, users are advised to use the more specialised :class:`ramanspy.Spectrum`,
:class:`ramanspy.SpectralImage` and :class:`ramanspy.SpectralVolume` classes, which extend the :class:`ramanspy.SpectralContainer` class
and thus inherit all features presented below.

Below, we will see how to define a spectral container, as well as how to use its main features.

.. seealso:: For more information about the :class:`ramanspy.Spectrum`, :class:`ramanspy.SpectralImage` and :class:`ramanspy.SpectralVolume`
             classes, check their documentation and/or the :ref:`Storing spectra`, :ref:`Storing imaging data` and :ref:`Storing volumetric data` tutorials respectively.
"""
import numpy as np
import ramanspy

# %%
# Initialisation
# """"""""""""""""""
# We can define :class:`ramanspy.SpectralContainer` containers of different dimensions by passing the corresponding intensity data
# and Raman wavenumber axis. For instance,

# an evenly spaced Raman wavenumber axis between 100 and 3000 cm^-1, consisting of 1500 elements.
spectral_axis = np.linspace(100, 3600, 1500)

# randomly generating intensity data array of shape (20, 1500)
spectral_data = np.random.rand(20, 1500)

# wrapping the data into a SpectralContainer instance
raman_object = ramanspy.SpectralContainer(spectral_data, spectral_axis)


# %%
# This can be any other shape, e.g.:
spectral_data = np.random.rand(1500)
raman_spectrum = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 1500)
raman_image = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 20, 1500)
raman_volume = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 20, 20, 1500)
raman_hypervolume = ramanspy.SpectralContainer(spectral_data, spectral_axis)

# %%
# If the spectral axis is in wavelength units (nm) and needs converting to wavenumber (cm :sup:`-1`), we can do that using
# the `wavelength_to_wavenumber` method within `ramanspy.utils`.

# %%
# We can also create a 2D :class:`ramanspy.SpectralContainer` container by stacking a collection of :class:`ramanspy.Spectrum` instances.
raman_spectra = [ramanspy.Spectrum(np.random.rand(1500), spectral_axis) for _ in range(5)]
raman_spectra_list = ramanspy.SpectralContainer.from_stack(raman_spectra)

raman_spectra_list.shape


# %%
# Features
# """"""""""""""
# Some of the main features and functionalities of the :class:`ramanspy.SpectralContainer` containers include:

# access to its spectral axis
raman_hypervolume.spectral_axis

# %%

# access to the length of the spectral axis
raman_hypervolume.spectral_length

# %%

# access to the shape of the data encapsulated within the instance
raman_hypervolume.spectral_data.shape

# %%

# access to the non-spectral (i.e. spatial) shape of the data encapsulated within the instance
raman_hypervolume.shape

# %%

# access to spatially collapsed data
raman_hypervolume.flat.shape

# %%

# access to a specific spectral band

raman_image.band(1500)
raman_spectrum.band(1500)


# %%
# Indexing
# """"""""""""""
# Another useful feature of the :class:`ramanspy.SpectralContainer` containers is their extensive spatial indexing capability.

print(type(raman_image[10, 10]))

# %%
# So, we can plot such indexed objects just as manually created ones:
raman_image[10, 10].plot()

# %%

print(type(raman_image[10]))

# %%

print(type(raman_volume[10]))

# %%

print(type(raman_hypervolume[10]))

# %%

print(type(raman_hypervolume[10, 10]))


# %%
# IO
# """"""""""""""
# :class:`ramanspy.SpectralContainer` containers (and thus subclasses) can also be saved as and loaded from pickle files.

# save
raman_image.save("my_raman_image")

# load
raman_image_ = ramanspy.SpectralContainer.load("my_raman_image")
raman_image_.shape
