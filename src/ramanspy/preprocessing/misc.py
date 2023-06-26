from numbers import Number
from typing import Tuple
import numpy as np

from . import PreprocessingStep
from ..core import Spectrum


class BackgroundSubtractor(PreprocessingStep):
    """
    Subtract a fixed reference background.\

    Parameters
    ----------
    background : Spectrum
        The reference background to subtract.
    """

    def __init__(self, *, background: Spectrum):
        super().__init__(_subtract_background, background=background)


class Cropper(PreprocessingStep):
    """
    Crop the intensity values and the shift axis associated with the band range(s) specified.

    Parameters
    ----------
    region : tuple of two elements
        The band intervals to crop (in cm^{-1}).
        Examples:
            [(None, 300)] - keeps the bands < 300cm-1
            [(3000, None)] - keeps the bands > 3000cm-1
            [(700, 1800)] - keeps the bands between 700 and 1800 (i.e. the "fingerprint" region)
    """

    def __init__(self, *, region: Tuple[Number or None, Number or None]):
        if len(region) != 2:
            raise ValueError("The region must be a tuple of two elements")

        super().__init__(_crop, region=region)


def _subtract_background(original_intensity_data, original_spectral_axis, background: Spectrum):
    if not np.array_equal(background.spectral_axis, original_spectral_axis):
        raise ValueError("The spectral axis of the background must match that of the spectral object to process")

    return original_intensity_data[..., :] - background.spectral_data, original_spectral_axis


def _crop(intensity_data, spectral_axis, region):
    indices_to_leave = _get_indices_to_leave(spectral_axis, region)

    return intensity_data[..., indices_to_leave], spectral_axis[indices_to_leave]


def _get_indices_to_leave(spectral_axis, region):
    start = spectral_axis[0] if region[0] is None else region[0]
    end = spectral_axis[-1] if region[1] is None else region[1]

    if start > end:
        # swap
        start, end = end, start

    indices_to_leave = np.logical_and(
        start <= spectral_axis, spectral_axis <= end)

    return indices_to_leave

