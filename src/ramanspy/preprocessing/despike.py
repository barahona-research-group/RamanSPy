import copy
from numbers import Number
import numpy as np

from . import PreprocessingStep


class WhitakerHayes(PreprocessingStep):
    """
    Cosmic rays removal based on modified z-scores filtering.

    Parameters
    ----------
    kernel_size : int, optional, default=3
        The size of the kernel to use for the algoritm.
    threshold : Number, optional, default=8
        The modified z_score threshold to use to identify spikes.

    References
    ----------
    Whitaker, D.A. and Hayes, K., 2018. A simple algorithm for despiking Raman spectra. Chemometrics and Intelligent Laboratory Systems, 179, pp.82-84.
    """

    def __init__(self, *, kernel_size: int = 3, threshold: Number = 8):
        super().__init__(_whitaker_hayes, kernel_size=kernel_size, threshold=threshold)


def _whitaker_hayes(intensity_data, spectral_axis, kernel_size, threshold):
    return np.apply_along_axis(_whitaker_hayes_spectrum, axis=-1, arr=intensity_data, kernel_size=kernel_size,
                               threshold=threshold), spectral_axis


def _whitaker_hayes_spectrum(intensity_values_array, kernel_size, threshold):
    spectrum_array = copy.deepcopy(intensity_values_array)

    spikes = _whitaker_hayes_modified_z_score(spectrum_array) > threshold

    while any(spike for spike in spikes if spike):
        changes = False

        for i in range(len(spikes)):
            if spikes[i]:
                neighbours = np.arange(max(0, i - kernel_size),
                                       min(len(spectrum_array) - 1, i + 1 + kernel_size))
                fixed_value = np.mean(spectrum_array[neighbours[spikes[neighbours] == 0]])

                if np.isnan(fixed_value):
                    continue

                spectrum_array[i] = fixed_value
                spikes[i] = 0
                changes = True

        if not changes:
            break

    return spectrum_array


def _modified_z_score(spectrum):
    """Calculates the modified z-scores of a given spectrum."""
    mad_term = np.median([np.abs(spectrum - np.median(spectrum))])
    modified_z_scores = np.array(0.6745 * (spectrum - np.median(spectrum)) / mad_term)

    return modified_z_scores


def _whitaker_hayes_modified_z_score(spectrum):
    """Calculates the Whitaker-Hayes modified z-scores of a given spectrum."""
    return np.abs(_modified_z_score(np.diff(spectrum)))
