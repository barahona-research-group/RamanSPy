from __future__ import annotations  # default if Python >= 3.10
import itertools
from numbers import Number
import os
import pickle
from typing import List, Union
import numpy as np
from scipy.signal import find_peaks

from . import plot
from . import utils


def _create_data(spectral_data, spectral_axis):
    if len(spectral_data.shape) == 1:
        return Spectrum(spectral_data, spectral_axis)
    elif len(spectral_data.shape) == 3:
        return SpectralImage(spectral_data, spectral_axis)
    elif len(spectral_data.shape) == 4:
        return SpectralVolume(spectral_data, spectral_axis)
    else:
        return SpectralContainer(spectral_data, spectral_axis)


class SpectralContainer:
    """
    The base class that settles as the backbone of the package. It encapsulates a spectral data
    container of an arbitrary dimension and defines relevant behaviour and information.

    Parameters
    ----------
    spectral_data : array_like of shape (x, y, z, ..., B)
        The intensity values to store. Last dimension must be the spectral dimension.
    spectral_axis : array_like of shape (B, )
        The Raman wavenumber axis (in cm\ :sup:`-1`). Order and length must match the last dimension of ``spectral_data``.


    .. note:: If your spectral data is not in Raman wavenumber units (cm\ :sup:`-1`) but in Raman wavelength (nm) instead,
              simply use the :meth:`ramanspy.utils.wavelength_to_wavenumber` method to convert your ``spectral_axis``
              before initialising a :class:`SpectralContainer` instance.

              Note that you will need to put in the excitation wavelength (nm) of the laser used to acquire the data of interest to make the conversion.

    Example
    ----------

    .. code::

        import numpy as np
        import ramanspy as rp

        spectral_data = np.random.rand(20, 1500)
        spectral_axis = np.linspace(100, 3600, 1500)

        # if the spectral axis is in wavelength units (nm) and needs converting
        spectral_axis = rp.utils.wavelength_to_wavenumber(spectral_axis)

        raman_object = rp.SpectralContainer(spectral_data, spectral_axis)
    """
    def __init__(self, spectral_data, spectral_axis):
        self.spectral_data = np.asarray(spectral_data, np.float32)
        self.spectral_axis = np.asarray(spectral_axis, np.float32)

        # Order data and axis by shift number values
        sorted_indices = self.spectral_axis.argsort()
        self.spectral_data = self.spectral_data[..., sorted_indices]
        self.spectral_axis = self.spectral_axis[sorted_indices]

        if self.spectral_data.shape[-1] != len(self.spectral_axis):
            raise ValueError(
                f"The last dimension of the data ({self.spectral_data.shape[-1]}) must match the axis provided ({len(self.spectral_axis)}).")

    def save(self, filename: str, directory: str = None):
        """
        Save the spectral object to a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to save the spectral object to.
        directory : str, optional
            The name of the directory to save the file in. Must be the full path to the directory or the path relative
            to the working directory. If not provided (default), the file will be saved in the working directory.
        """
        full_filename = os.path.join(directory, filename) if directory is not None else filename
        with open(full_filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load a spectral object from a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to load a spectral object from. Must be the full path or the path relative to the working directory.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_stack(cls, stack: List[Spectrum]) -> SpectralContainer:
        """
        Returns the combined Raman object defined by stacking the collection of individual spectra given.

        The spectral axes of the spectra provided must match.
        """
        if not utils.is_aligned(stack):
            ValueError("Cannot stack unaligned spectral objects. Spectral axes must match.")

        return cls(np.vstack([obj.flat.spectral_data for obj in stack]), stack[0].spectral_axis)

    @property
    def flat(self) -> SpectralContainer:
        """
        Flatten all spatial dimensions of the spectral object into a single one.

        Returns
        -------
        numpy.ndarray of shape (dim_1*dim_2*...*dim_n, B)
        """
        return SpectralContainer(self.spectral_data.reshape(-1, self.spectral_length), self.spectral_axis)

    @property
    def shape(self) -> tuple[int]:
        """
        Returns the (spatial) shape of the spectral object (i.e. without the last dimension).
        """
        return self.spectral_data.shape[:-1] if len(self.spectral_data.shape) >= 2 else (1,)

    @property
    def spectral_length(self) -> int:
        """
        Returns the spectral length B of the spectral object.
        """
        return len(self.spectral_axis)

    @property
    def mean(self) -> Spectrum:
        """
        Returns the mean spectrum in the spectral object.
        """
        return Spectrum(np.mean(self.flat.spectral_data, axis=0), self.spectral_axis)

    # TODO: spatial vs spectral indexing
    def __getitem__(self, key):
        if self.shape == (1,):
            raise ValueError(
                "Only spatial indexing is supported. To index spectrally, use the ramanspy.preprocessing.misc.Cropper class.")

        spectral_data_slice = self.spectral_data[key]

        if len(spectral_data_slice.shape) == 0:
            return spectral_data_slice
        else:
            return _create_data(spectral_data_slice, self.spectral_axis)

    def band(self, spectral_band: Number) -> np.ndarray:
        """Returns a spectral slice across the closest spectral band in the axis to the one given."""

        # Check if band given is within the spectral axis
        min_band = self.spectral_axis.min()
        max_band = self.spectral_axis.max()
        if not (min_band <= spectral_band <= max_band):
            raise ValueError(
                f"Band ({spectral_band}) must be within the bounds of the spectral axis ([{min_band}, {max_band}])")

        # Find the closest band in the axis to the one provided
        closest_band_index = np.argmin(np.abs(self.spectral_axis - spectral_band))

        # Return the slice across that band
        return self.spectral_data[..., closest_band_index]


class Spectrum(SpectralContainer):
    """
    The :class:`Spectrum` class defines a 1D spectroscopic signal of an arbitrary spectral length.

    Example
    ----------

    .. code::

        import numpy as np
        import ramanspy as rp

        spectral_data = np.random.rand(1500)
        spectral_axis = np.linspace(100, 3600, 1500)

        raman_spectrum = rp.Spectrum(spectral_data, spectral_axis)
    """

    def plot(self, **kwargs):
        """
        Plots the spectrum.

        Parameters
        ----------
        **kwargs : keyword arguments, optional,
            Check the :meth:`ramanspy.plot.spectra' method for a list of keyword parameters.
        """
        return plot.spectra(self, **kwargs)

    def peaks(self,
              *,
              height=None,
              threshold=None,
              distance=None,
              prominence=None,
              width=None,
              wlen=None,
              rel_height=0.5,
              plateau_size=None,
              ):

        peaks, properties = find_peaks(self.spectral_data,  height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
        return peaks, properties


class SpectralImage(SpectralContainer):
    """
    The :class:`SpectralImage` class defines a 2D spectroscopic image. Dimensions must be in the order of
    (x, y, B).

    Example
    ----------
    
    .. code::

        import numpy as np
        import ramanspy as rp

        spectral_data = np.random.rand(50, 50, 1500)
        spectral_axis = np.linspace(100, 3600, 1500)

        raman_image = rp.SpectralImage(spectral_data, spectral_axis)
    """

    def plot(self, bands: Union[Number, List[Number]], **kwargs):
        """
        Plots the spectral image slice(s) across the spectral image, defined by the band(s) provided (using the closest
        band(s) in the spectral axis of the image to the one(s) given).

        Parameters
        ----------
        bands : Number or List[Number]
            The spectral bands to plot across.
        **kwargs : keyword arguments, optional,
            Check the :meth:`ramanspy.plot.image' method for a list of keyword parameters.
        """
        if isinstance(bands, Number):
            bands = [bands]

        spectral_slices = [self.band(band) for band in bands]
        kwargs['cbar_label'] = [f"{cbar_label} ({band} cm$^{{{-1}}}$)" for cbar_label, band in
                                zip(itertools.repeat(kwargs.pop('cbar_label', 'Peak intensity'), len(spectral_slices)), bands)]

        return plot.image(spectral_slices, **kwargs)


class SpectralVolume(SpectralContainer):
    """
    The :class:`SpectralVolume` class defines a 3D spectroscopic volume. Dimensions must be in the order of
    (x, y, z, B).

    Example
    ----------

    .. code::

        import numpy as np
        import ramanspy as rp

        spectral_data = np.random.rand(50, 50, 10, 1500)
        spectral_axis = np.linspace(100, 3600, 1500)

        raman_volume = rp.SpectralVolume(spectral_data, spectral_axis)
    """

    @classmethod
    def from_image_stack(cls, image_stack: List[SpectralImage]) -> SpectralVolume:
        """
        Returns the volumetric Raman object defined by z-stacking the collection of spectral images given.

        All dimensions of the spectral images must match, as well as their spectral axes.
        """
        if not utils.is_aligned(image_stack):
            ValueError("Cannot create a spectral volume out of unaligned spectral images. Spectral axes must match.")

        return cls(np.dstack([image.spectral_data[..., np.newaxis, :] for image in image_stack]),
                   image_stack[0].spectral_axis)

    def plot(self, bands, **kwargs):
        """
        Plots the spectral volume slice(s) across the spectral volume, defined by the band(s) provided (using the closest
        band(s) in the spectral axis of the image to the one(s) given).

        Parameters
        ----------
        bands : Number or List[Number]
            The spectral bands to plot across.
        **kwargs : keyword arguments, optional,
            Check the :meth:`ramanspy.plot.volume' method for a list of keyword parameters.
        """
        if isinstance(bands, Number):
            bands = [bands]

        spectral_slices = [self.band(band) for band in bands]
        kwargs['cbar_label'] = [f"{cbar_label} ({band} cm$^{{{-1}}}$)" for cbar_label, band in
                                zip(itertools.repeat(kwargs.pop('cbar_label', 'Peak intensity'), len(spectral_slices)), bands)]

        return plot.volume(spectral_slices, **kwargs)

    def layer(self, layer_index: int) -> SpectralImage:
        """Returns the :class:`SpectralImage` layer specified by the given index as a SpectralImage. Index must be between 0 and |z dimension|-1."""
        if not (0 <= layer_index <= self.shape[-1] - 1):
            ValueError(
                f"The layer index must be between 0 and {self.shape[-1] - 1} inclusively. Got {layer_index} instead.")

        return SpectralImage(self.spectral_data[..., layer_index, :], self.spectral_axis)


# for typing
SpectralObject = Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume]
