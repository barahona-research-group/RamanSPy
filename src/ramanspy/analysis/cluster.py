import numpy as np
import sklearn.cluster as cluster

from .Step import AnalysisStep


class KMeans(AnalysisStep):
    """
    k-means clustering.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    **kwargs :
        Check original `implementation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ for additional parameters.


    .. note :: Implementation and documentation based on `scikit-learn <https://scikit-learn.org>`_.
    """

    def __init__(self, *, n_clusters, **kwargs):
        super().__init__(_kmeans, n_clusters, **kwargs)


def _kmeans(original_intensity_data, n_clusters, **kwargs):
    kmeans = cluster.KMeans(n_clusters, **kwargs)

    spectra_clustered = kmeans.fit_predict(original_intensity_data)

    # Convert to one-hot encodings to represent an appropriate abundance-like map
    n_values = np.max(spectra_clustered) + 1
    spectra_clustered = np.eye(n_values)[spectra_clustered]

    return spectra_clustered, kmeans.cluster_centers_
