from numbers import Number
import numpy as np

from . import PreprocessingStep


class Vector(PreprocessingStep):
    """
    Vector normalisation.

    Applied to each spectrum individually.

    .. math::

        x = x/||x||_{2}

    Parameters
    ----------
    pixelwise : bool
        If ``True`` (default), method is applied to each spectrum individually. If ``False``, spectra are divided by the norm of
        the spectra with the largest norm in the given :class:`ramanspy.SpectralContainer` instance.
    """

    def __init__(self, *, pixelwise: bool = True):
        super().__init__(_vector_norm, pixelwise=pixelwise)


class MinMax(PreprocessingStep):
    """
    Min-max normalisation.

    Scales the data to the interval [a, b].

    .. math::

        x = (a + (x-min(x))*(b-a))/(max(x) - min(x))

    Parameters
    ----------
    pixelwise : bool
        If ``True`` (default), method is applied to each spectrum individually. If ``False``, the global minimum and maximum is used.
    a : Number, optional, default=0
    b : Number, optional, default=1
    """

    def __init__(self, *, pixelwise: bool = True, a: Number = 0, b: Number = 1):
        super().__init__(_minmax_norm, pixelwise=pixelwise, a=a, b=b)


class MaxIntensity(PreprocessingStep):
    """
    Max intensity normalisation.

    .. math::

        x = x/max(x)

    Parameters
    ----------
    pixelwise : bool
        If ``True`` (default), method is applied to each spectra individually. If ``False``, spectra are divided by the global maximum intensity.
    """

    def __init__(self, *, pixelwise: bool = True):
        super().__init__(_max_intensity_norm, pixelwise=pixelwise)


class AUC(PreprocessingStep):
    """
    Area under the curve normalisation.

    .. math::

        x = x/AUC(x)

    Parameters
    ----------
    pixelwise : bool
        If ``True`` (default), method is applied to each spectra individually. If ``False``, spectra are divided by the area under
        the curve of the spectrum with the largest one in the given :class:`ramanspy.SpectralContainer` instance.
    """

    def __init__(self, *, pixelwise: bool = True):
        super().__init__(_auc_norm, pixelwise=pixelwise)


def _vector_norm(intensity_data, spectral_axis, *, pixelwise: bool):
    vector_norms = np.linalg.norm(intensity_data, axis=-1, keepdims=True)

    if pixelwise:
        return intensity_data[..., :] / vector_norms, spectral_axis
    else:
        return intensity_data / vector_norms.max(), spectral_axis


def _minmax_norm(intensity_data, spectral_axis, a, b, *, pixelwise: bool):
    if pixelwise:
        mins = np.min(intensity_data, axis=-1, keepdims=True)
        maxs = np.max(intensity_data, axis=-1, keepdims=True)

        return a + (intensity_data[..., :] - mins) * (b - a) / (maxs - mins), spectral_axis

    else:
        mins = np.min(intensity_data)
        maxs = np.max(intensity_data)

        return a + (intensity_data - mins) * (b - a) / (maxs - mins), spectral_axis


def _max_intensity_norm(intensity_data, spectral_axis, *, pixelwise: bool):
    if pixelwise:
        return intensity_data[..., :] / np.max(intensity_data, axis=-1, keepdims=True), spectral_axis
    else:
        return intensity_data / np.max(intensity_data), spectral_axis


def _auc_norm(intensity_data, spectral_axis, *, pixelwise: bool):
    aucs = np.trapz(intensity_data, spectral_axis, axis=-1)[..., np.newaxis]

    if pixelwise:
        return intensity_data[..., :] / aucs, spectral_axis
    else:
        return intensity_data / aucs.max(), spectral_axis
