import numpy as np
import scipy.linalg as splin
import functools
from typing import Literal
import pysptools.abundance_maps.amaps as amaps
import pysptools.eea as eea
from pysptools.eea import nfindr

from .Step import AnalysisStep


"""
List of the available methods for calculating fractional abundances.
"""
abundance_methods = {
    'ucls': amaps.UCLS,
    'nnls': amaps.NNLS,
    'fcls': amaps.FCLS,
}


def unmixer(endmember_func):
    @functools.wraps(endmember_func)
    def wrap(spectral_data, n_endmembers, abundance_method):
        endmembers = endmember_func(spectral_data, n_endmembers)

        abundance_method_ = abundance_methods.get(abundance_method, None)
        if abundance_method is None:
            raise ValueError(
                f"{abundance_method} is not a valid abundance method. Possible methods are {abundance_methods.keys()}")

        abundances = abundance_method_(spectral_data, endmembers)

        return abundances, endmembers

    return wrap


class PPI(AnalysisStep):
    """
    Pixel Purity Index (PPI).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Boardman, J.W., Kruse, F.A. and Green, R.O., 1995. Mapping target signatures via partial unmixing of AVIRIS data.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(eea.PPI().extract), n_endmembers, abundance_method)


class FIPPI(AnalysisStep):
    """
    Fast Iterative Pixel Purity Index (FIPPI).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Chang, C.I. and Plaza, A., 2006. A fast iterative algorithm for implementation of pixel purity index. IEEE Geoscience and Remote Sensing Letters, 3(1), pp.63-67.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(eea.FIPPI().extract), n_endmembers, abundance_method)


class NFINDR(AnalysisStep):
    """
    N-FINDR.

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Winter, M.E., 1999, October. N-FINDR: An algorithm for fast autonomous spectral end-member determination in hyperspectral data. In Imaging Spectrometry V (Vol. 3753, pp. 266-275). SPIE.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_nfindr), n_endmembers, abundance_method)


class VCA(AnalysisStep):
    """
    Vertex Component Analysis (VCA).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `the MATLAB code provided by the authors <http://www.lx.it.pt/~bioucas/code.htm>`_
               and `Adrien Lagrange's translation to Python <https://github.com/Laadr/VCA>`_.


    References
    ----------
    Nascimento, J.M. and Dias, J.M., 2005. Vertex component analysis: A fast algorithm to unmix hyperspectral data. IEEE transactions on Geoscience and Remote Sensing, 43(4), pp.898-910.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_vca), n_endmembers, abundance_method)


def _vca(data, n_endmembers):
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_centered = data - data_mean

    # Project mean-centered data to n_endmembers subspace
    Ud = splin.svd(np.dot(data_centered, data_centered.T) / float(data.shape[0]))[0]
    x_p = np.dot(Ud[:, :n_endmembers].T, data_centered)

    # Estimate SNR
    P_y = np.sum(data.T ** 2) / float(data.shape[0])
    P_x = np.sum(x_p ** 2) / float(data.shape[0]) + np.sum(data_mean ** 2)
    SNR = 10 * np.log10((P_x - x_p.shape[0] / data.shape[1] * P_y) / (P_y - P_x))

    SNR_threshold = 15 + 10 * np.log10(n_endmembers)
    if SNR < SNR_threshold:
        # Project to n_endmembers-1 subspace
        # """"""""""""""""""""""""""""""""""""""
        Yp = np.dot(Ud[:, :n_endmembers - 1], x_p[:n_endmembers - 1, :]) + data_mean

        x = x_p[:n_endmembers - 1, :]
        c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, data.shape[0]))))
    else:
        # Projective Projection
        # """""""""""""""""""""""
        Ud = splin.svd(np.dot(data.T, data) / float(data.shape[0]))[0][:, :n_endmembers]

        x = np.dot(Ud.T, data.T)
        Yp = np.dot(Ud, x[:n_endmembers, :])

        u = np.mean(x, axis=1, keepdims=True)
        y = x / np.dot(u.T, x)

    # VCA algorithm
    # --------------
    indice = np.zeros(n_endmembers, dtype=int)
    A = np.zeros((n_endmembers, n_endmembers))
    A[-1, 0] = 1

    for i in range(n_endmembers):
        w = np.random.rand(n_endmembers, 1)
        f = w - np.dot(A, np.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = np.dot(f.T, y)

        indice[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indice[i]]

    Ae = Yp[:, indice]

    return Ae.T


def _nfindr(spectral_data, num_of_endmembers):
    endmembers, _, _, _ = eea.nfindr.NFINDR(spectral_data, num_of_endmembers)
    return endmembers
