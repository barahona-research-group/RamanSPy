from numbers import Number
from typing import List, Tuple
import numpy as np
import pybaselines

from . import PreprocessingStep


"""
List of the available methods for spectral baseline correction offered by the package.
"""


class PybaselinesCorrector(PreprocessingStep):
    """
    A class that wraps pybaselines baseline correction function and converts them to an executable :class:`.Preprocessing` object.
    """

    @staticmethod
    def _pybaselines_baseline_extractor(func):
        # Wrapper that extracts the baseline result and discard the params result from the given pybaselines method.
        def wrap(*args, **kwargs):
            baseline, params = func(*args, **kwargs)
            return baseline

        return wrap

    @staticmethod
    def _pybaselines_broadcaster(func):
        # Wrapper that applies the given pybaselines function over the n-dimensional data provided.
        def wrap(intensity_data, spectral_axis, **kwargs):
            extractor_func = PybaselinesCorrector._pybaselines_baseline_extractor(func)
            baseline_data = np.apply_along_axis(extractor_func, axis=-1, arr=intensity_data, x_data=spectral_axis, **kwargs)

            preprocessed_spectral_data = intensity_data - baseline_data

            return preprocessed_spectral_data, spectral_axis

        return wrap

    def __init__(self, pybaselines_function, **kwargs):
        # Using the wrappers in the class, create a PreprocessingStep object that applies the pybaselines function provided.

        broadcastable_pybaselines_function = PybaselinesCorrector._pybaselines_broadcaster(pybaselines_function)
        super().__init__(broadcastable_pybaselines_function, **kwargs)


class ASLS(PybaselinesCorrector):
    """
    Baseline correction based on Asymmetric Least Squares (AsLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.asls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 2005, 1(1).
    """
    def __init__(self, *, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        super().__init__(pybaselines.whittaker.asls, lam=lam, p=p, diff_order=diff_order,  max_iter=max_iter,
                         tol=tol, weights=weights)


class IASLS(PybaselinesCorrector):
    """
    Baseline correction based on Improved Asymmetric Least Squares (IAsLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.iasls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    He, S., et al. Baseline correction for raman spectra using an improved asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.
    """

    def __init__(self, *, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3, weights=None, diff_order=2):
        super().__init__(pybaselines.whittaker.iasls, lam=lam, p=p, lam_1=lam_1, max_iter=max_iter, tol=tol, weights=weights, diff_order=diff_order)


class AIRPLS(PybaselinesCorrector):
    """
    Baseline correction based on Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.airpls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.
    """

    def __init__(self, *, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        super().__init__(pybaselines.whittaker.airpls, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights)


class ARPLS(PybaselinesCorrector):
    """
    Baseline correction based on Asymmetrically Reweighted Penalized Least Squares (arPLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.arpls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted penalized least squares smoothing. Analyst, 2015, 140, 250-257.
    """

    def __init__(self, *, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        super().__init__(pybaselines.whittaker.arpls, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights)


class DRPLS(PybaselinesCorrector):
    """
    Baseline correction based on Doubly Reweighted Penalized Least Squares (drPLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.drpls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted penalized least squares, Applied Optics, 2019, 58, 3913-3920.
    """

    def __init__(self, *, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None, diff_order=2):
        super().__init__(pybaselines.whittaker.drpls, lam=lam, eta=eta, max_iter=max_iter, tol=tol, weights=weights, diff_order=diff_order)


class IARPLS(PybaselinesCorrector):
    """
    Baseline correction based on Improved Asymmetrically Reweighted Penalized Least Squares (IarPLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.iarpls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically reweighted penalized least squares for Raman spectrum. Applied Optics, 2020, 59, 10933-10943.
    """

    def __init__(self, *, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        super().__init__(pybaselines.whittaker.iarpls, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights)


class ASPLS(PybaselinesCorrector):
    """
    Baseline correction based on Adaptive Smoothness Penalized Least Squares (asPLS).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.aspls>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using adaptive smoothness parameter penalized least squares method.  Spectroscopy Letters, 2020, 53(3), 222-233.
    """

    def __init__(self, *, lam=1e5, diff_order=2, max_iter=100, tol=1e-3, weights=None, alpha=None):
        super().__init__(pybaselines.whittaker.aspls, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights, alpha=alpha)


class Poly(PybaselinesCorrector):
    """
    Baseline correction based on polynomial fitting.

    Parameters
    ----------
    poly_order : int, optional
        Order of polynomial to fit. Default is 2.
    regions : list of tuples, optional
        Describes the regions (in cm^{-1}) used for selective masking. If ``None`` (default), all points are used.
        Examples:
            [(None, 300)] - only uses the bands < 300cm-1
            [(3000, None)] - only uses the bands > 3000cm-1
            [(700, 1800)] - only uses the bands between 700 and 1800 (i.e. the "fingerprint" region)
            [(300, 400), (500, 600)] - only uses the bands between 300 and 400, and 500 and 600


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.
    """
    def __init__(self, *, poly_order=2, regions: List[Tuple[Number or None, Number or None]] = None):
        if regions is not None:
            for region in regions:
                if len(region) != 2:
                    raise ValueError("Region must be a list of tuples of two elements")

        super().__init__(pybaselines.polynomial.poly, poly_order=poly_order, weights=regions)

    def _get_mask(self, spectral_axis):
        mask = np.zeros_like(spectral_axis)

        for region in self.kwargs['weights']:
            start = spectral_axis[0] if region[0] is None else region[0]
            end = spectral_axis[-1] if region[1] is None else region[1]

            if start > end:
                # swap
                start, end = end, start

            ones = np.logical_and(start <= spectral_axis, spectral_axis <= end)

            mask[ones] = 1

        return mask

    def __call__(self, spectral_data, spectral_axis, **kwargs):
        # use selective masking if regions are provided
        weights = self._get_mask(spectral_axis) if self.kwargs['weights'] is not None else None

        # exchange weights in kwargs with the mask
        kwargs = {k: v for k, v in self.kwargs.items() if k != 'weights'}
        kwargs['weights'] = weights

        return self.method(spectral_data, spectral_axis, **kwargs)


class ModPoly(PybaselinesCorrector):
    """
    Baseline correction based on modified polynomial fitting.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.polynomial.modpoly>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Lieber, C., et al. Automated method for subtraction of fluorescence from biological raman spectra. Applied Spectroscopy, 2003, 57(11), 1363-1367.

    Gan, F., et al. Baseline correction by improved iterative polynomial fitting with automatic threshold. Chemometrics and Intelligent Laboratory Systems, 2006, 82, 59-65.
    """
    def __init__(self, *, poly_order=2, tol=0.001, max_iter=250, weights=None, use_original=False, mask_initial_peaks=False):
        super().__init__(pybaselines.polynomial.modpoly, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights, use_original=use_original, mask_initial_peaks=mask_initial_peaks)


class PenalisedPoly(PybaselinesCorrector):
    """
    Baseline correction based on penalised polynomial fitting.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.polynomial.penalized_poly>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.
    """
    def __init__(self, *, poly_order=2, tol=0.001, max_iter=250, weights=None, cost_function='asymmetric_truncated_quadratic', threshold=None, alpha_factor=0.99):
        super().__init__(pybaselines.polynomial.penalized_poly, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights, cost_function=cost_function, threshold=threshold, alpha_factor=alpha_factor)


class IModPoly(PybaselinesCorrector):
    """
    Baseline correction based on improved modified polynomial fitting.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.polynomial.imodpoly>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Zhao, J., et al. Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy, 2007, 61(11), 1225-1232.
    """
    def __init__(self, *, poly_order=2, tol=0.001, max_iter=250, weights=None, use_original=False, mask_initial_peaks=True, num_std=1):
        super().__init__(pybaselines.polynomial.imodpoly, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights, use_original=use_original, mask_initial_peaks=mask_initial_peaks, num_std=num_std)


class Goldindec(PybaselinesCorrector):
    """
    Baseline correction based on Goldindec.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.polynomial.goldinec>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline Correction. Applied Spectroscopy, 2015, 69(7), 834-842.
    """
    def __init__(self, *, poly_order=2, tol=0.001, max_iter=250, weights=None, cost_function='asymmetric_indec', peak_ratio=0.5, alpha_factor=0.99, tol_2=0.001, tol_3=1e-06, max_iter_2=100):
        super().__init__(pybaselines.polynomial.goldindec, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights, cost_function=cost_function, peak_ratio=peak_ratio, alpha_factor=alpha_factor, tol_2=tol_2, tol_3=tol_3, max_iter_2=max_iter_2)


class IRSQR(PybaselinesCorrector):
    """
    Baseline correction based on Iterative Reweighted Spline Quantile Regression (IRSQR).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.spline.irsqr>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Han, Q., et al. Iterative Reweighted Quantile Regression Using Augmented Lagrangian Optimization for Baseline Correction. 2018 5th International Conference on Information Science and Control Engineering (ICISCE), 2018, 280-284.
    """
    def __init__(self, *, lam=100, quantile=0.05, num_knots=100, spline_degree=3, diff_order=3, max_iter=100, tol=1e-06, weights=None, eps=None):
        super().__init__(pybaselines.spline.irsqr, lam=lam, quantile=quantile, num_knots=num_knots, spline_degree=spline_degree, diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights, eps=eps)


class CornerCutting(PybaselinesCorrector):
    """
    Baseline correction based on Corner Cutting.

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.spline.corner_cutting>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline Construction. Analyst, 2015, 140(23), 7984-7996.
    """
    def __init__(self, *, max_iter=100):
        super().__init__(pybaselines.spline.corner_cutting, max_iter=max_iter)


class FABC(PybaselinesCorrector):
    """
    Baseline correction based on Fully automatic baseline correction (FABC).

    Parameters
    ----------
    **kwargs :
        Check original `implementation <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.classification.fabc>`_ for information about parameters.


    .. note :: Implementation based on `pybaselines <https://pybaselines.readthedocs.io>`_.


    References
    ----------
    Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline Construction. Analyst, 2015, 140(23), 7984-7996.
    """
    def __init__(self, *, lam=1000000.0, scale=None, num_std=3.0, diff_order=2, min_length=2, weights=None, weights_as_mask=False, x_data=None, **pad_kwargs):
        super().__init__(pybaselines.classification.fabc, lam=lam, scale=scale, num_std=num_std, diff_order=diff_order, min_length=min_length, weights=weights, weights_as_mask=weights_as_mask, x_data=x_data, **pad_kwargs)