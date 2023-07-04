import functools
import sklearn.decomposition as decomp

from .Step import AnalysisStep


class PCA(AnalysisStep):
    """
    Principal component analysis (PCA).


    Parameters
    ----------
    n_components : int, float or 'mle'
        The number of components.
    **kwargs :
        See original `implementation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ for additional parameters.


    .. note :: Implementation and documentation based on`scikit-learn <https://scikit-learn.org>`_.
    """
    def __init__(self, *, n_components, **kwargs):
        super().__init__(scikit_learn_wrapper(decomp.PCA), n_components, **kwargs)


class ICA(AnalysisStep):
    """
    Independent component analysis (ICA).

    Parameters
    ----------
    n_components : int
        The number of components.
    **kwargs :
        Check original `implementation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.ICA.html>`_ for additional parameters.


    .. note :: Implementation and documentation based on`scikit-learn <https://scikit-learn.org>`_.
    """
    def __init__(self, *, n_components, **kwargs):
        super().__init__(scikit_learn_wrapper(decomp.FastICA), n_components, **kwargs)


class NMF(AnalysisStep):
    """
    Non-negative matrix factorisation (NMF).

    Data must be non-negative. If negative values are present, one can use :class:`ramanspy.preprocessing.normalise.MinMax` to scale the data.

    Parameters
    ----------
    n_components : int
        The number of components.
    **kwargs :
        Check original `implementation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html>`_ for additional parameters.


    .. note :: Implementation and documentation based on`scikit-learn <https://scikit-learn.org>`_.
    """

    def __init__(self, *, n_components, **kwargs):
        super().__init__(scikit_learn_wrapper(decomp.NMF), n_components, **kwargs)


def scikit_learn_wrapper(scikit_learn_model):
    @functools.wraps(scikit_learn_model)
    def wrap(data, *args, **kwargs):
        model = scikit_learn_model(*args, **kwargs)
        spectra_transformed = model.fit_transform(data)

        return spectra_transformed, model.components_

    return wrap
