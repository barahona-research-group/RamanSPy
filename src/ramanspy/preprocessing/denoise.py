from numbers import Number
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

from . import PreprocessingStep


class SavGol(PreprocessingStep):
    """
    Denoising based on Savitzky-Golay filtering.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ for information about parameters.


    .. note :: Implementation based on `scipy <https://docs.scipy.org/doc/scipy/>`_.


    References
    ----------
    Savitzky, A. and Golay, M.J., 1964. Smoothing and differentiation of data by simplified least squares procedures. Analytical chemistry, 36(8), pp.1627-1639.
    """

    def __init__(self, *, window_length, polyorder, deriv=0, delta=1.0, mode='interp', cval=0.0):
        super().__init__(_savgol, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode, cval=cval)


class Gaussian(PreprocessingStep):
    """
    Denoising based on a Gaussian filter.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_ for information about parameters.


    .. note :: Implementation based on `scipy <https://docs.scipy.org/doc/scipy/>`_.
    """

    def __init__(self, *, sigma=1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=None):
        super().__init__(_gauss, sigma=sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate, radius=radius)


# TODO: Check implementation
class Whittaker(PreprocessingStep):
    """
    Denoising based on Discrete Penalised Least Squares (a.k.a Whittaker−Henderson smoothing).

    Parameters
    ----------
    lam : Number, optional, default=1e3
        Hyperparameter that controls the smoothing.
    d : int, optional, default=1
        Smoothing order.

    References
    ----------
    Eilers, P.H., 2003. A perfect smoother. Analytical chemistry, 75(14), pp.3631-3636.
    """

    def __init__(self, *, lam: Number = 1e3, d: int = 2):
        super().__init__(_whittaker, lam=lam, d=d)


class Kernel(PreprocessingStep):
    """
    Denoising based on kernel/window smoothers.

    Parameters
    ----------
    method : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        The type of kernel to use.
    kernel_size : int, optional, default=11
        The size of the window/kernel to use.
    """

    def __init__(self, *, method, kernel_size: int = 11):
        super().__init__(_kernel, method=method, kernel_size=kernel_size)


def _savgol(intensity_data, spectral_axis, **kwargs):
    return savgol_filter(intensity_data, **kwargs), spectral_axis


def _whittaker(intensity_data, spectral_axis, lam, d):
    I = np.eye(intensity_data.shape[-1])
    D = np.diff(I, d).T

    return intensity_data @ np.linalg.inv(I + lam * D.T @ D).T, spectral_axis


def _kernel(intensity_data, spectral_axis, method, kernel_size):
    return np.apply_along_axis(_kernel_spectrum, axis=-1, arr=intensity_data, method=method, kernel_size=kernel_size), spectral_axis


def _kernel_spectrum(intensity_values_array, method, kernel_size):
    """
    Adapted from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth
    """
    # adding padding
    padded_spectrum = np.r_[
        intensity_values_array[kernel_size - 1:0:-1], intensity_values_array, intensity_values_array[
                                                                                   -2:-kernel_size - 1:-1]]

    if method == 'flat':
        kernel_window = np.ones(kernel_size, 'd')
    else:
        kernel_window = getattr(np, method)(kernel_size)

    denoised_spectrum = np.convolve(kernel_window / kernel_window.sum(), padded_spectrum, mode='valid')

    denoised_spectrum = denoised_spectrum[int(kernel_size / 2):-int(kernel_size / 2)].astype(
        np.float32)  # TODO: check conversions

    return denoised_spectrum


def _gauss(intensity_data, spectral_axis, **kwargs):
    return gaussian_filter(intensity_data, **kwargs), spectral_axis
