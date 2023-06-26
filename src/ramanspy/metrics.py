import numpy as np

from .core import Spectrum


def check_compatibility(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray):
    if type(a) != type(b):
        raise TypeError(f"The types of the spectra must match ({type(a)} != {type(b)}).")
    if isinstance(a, Spectrum) and isinstance(b, Spectrum):
        if a.spectral_data.shape != b.spectral_data.shape:
            raise ValueError(f"The shape of the spectra must match ({a.shape} != {b.shape}).")
        if not np.array_equal(a.spectral_axis, b.spectral_axis):
            raise ValueError(f"The wavelengths of the spectra must match ({a.spectral_axis} != {b.spectral_axis}).")
    else:
        if a.shape != b.shape:
            raise ValueError(f"The shape of the spectra must match ({a.shape} != {b.shape}).")


def spectral_metric(metric_func):
    def wrap(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray, *args, **kwargs):
        check_compatibility(a, b)

        if isinstance(a, Spectrum) and isinstance(b, Spectrum):
            return metric_func(a.spectral_data, b.spectral_data, *args, **kwargs)
        else:
            return metric_func(a, b, *args, **kwargs)

    wrap.__doc__ = metric_func.__doc__

    return wrap


@spectral_metric
def MAE(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray):
    """
    Mean Absolute Error (MAE).

    MAE is an Euclidean distance-based measure equal to the average of the absolute errors. The smaller the MAE, the more similar the two spectra are.

    .. math::

        MAE = \\frac{1}{n} \\sum_{i=1}^{n} |a_i - b_i|

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        rp.metrics.MAE(spectrum_1, spectrum_2)
    """
    return np.mean(np.abs(a - b))


@spectral_metric
def MSE(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray):
    """
    Mean Squared Error (MSE).

    MSE is an Euclidean distance-based measure equal to the average of the square of the errors. The smaller the MSE, the more similar the two spectra are.

    .. math::

        MSE = \\frac{1}{n} \\sum_{i=1}^{n} (a_i - b_i)^2

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        rp.metrics.MSE(spectrum_1, spectrum_2)
    """
    return np.mean((a - b) ** 2)


@spectral_metric
def RMSE(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray):
    """
    Root-mean-square error (RMSE).

    The RMSE is the square root of the MSE. The smaller the RMSE, the more similar the two spectra are.

    .. math::

        RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (a_i - b_i)^2}

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        rp.metrics.RMSE(spectrum_1, spectrum_2)
    """
    return np.sqrt(MSE(a, b))


@spectral_metric
def SAD(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray):
    """
    Spectral Angle Distance (SAD).

    The SAD is the angle between two spectra in the n-dimensional space. The smaller the SAD, the more similar the two spectra are.

    .. math::

        SAD = \\arccos \\left( \\frac{a \\cdot b}{||a|| \\cdot ||b||} \\right)

    References
    -----------
    Kruse, F.A., Lefkoff, A.B., Boardman, J.W., Heidebrecht, K.B., Shapiro, A.T., Barloon, P.J. and Goetz, A.F.H., 1993. The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data. Remote sensing of environment, 44(2-3), pp.145-163.


    Examples:
    ----------

    .. code::

        import ramanspy as rp

        rp.metrics.SAD(spectrum_1, spectrum_2)
    """
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos = np.clip(cos, -1, 1)
    return np.arccos(cos)


@spectral_metric
def SID(a: Spectrum or np.ndarray, b: Spectrum or np.ndarray, *, epsilon=1e-6):
    """
    Spectral Information Divergence (SID).

    The SID is a information-theoretic measure of the difference between two spectra based on Kullback-Leibler divergence.
    The smaller the SID, the more similar the two spectra are.

    .. math::

        SID = D_{KL}(a||b) + D_{KL}(b||a),


    where

    .. math::

        D_{KL}(p||q) = \\sum_{i=1}^{n} p_i \\log \\frac{p_i}{q_i}

    References
    -----------
    Chang, C.I., 1999, June. Spectral information divergence for hyperspectral image analysis. In IEEE 1999 International Geoscience and Remote Sensing Symposium. IGARSS'99 (Cat. No. 99CH36293) (Vol. 1, pp. 509-511). IEEE.

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        rp.metrics.SID(spectrum_1, spectrum_2)
    """
    # Offset if intensity values are smaller than 0 (SID is invariant to translation)
    a_min = np.min(a)
    b_min = np.min(b)

    a_data = a + (-1) * a_min if a_min < 0 else a
    b_data = b + (-1) * b_min if b_min < 0 else b

    # convert to probabilities
    p = a_data / a_data.sum() + epsilon
    q = b_data / b_data.sum() + epsilon

    # calculate KL divergences
    d1 = np.sum(p * np.log(p / q))
    d2 = np.sum(q * np.log(q / p))

    return d1 + d2
