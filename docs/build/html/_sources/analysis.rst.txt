=================================
Analysis
=================================
.. currentmodule:: ramanspy.analysis

`RamanSPy` provides a number of built-in methods for spectral analysis, including decomposition, clustering and spectral unmixing.
Similarly to the preprocessing methods, these are built into `RamanSPy` as standardised classes and can thus be readily accessed
and applied to any type of Raman spectroscopic data loaded into the framework.

Analysis functionality is given within the :mod:`ramanspy.analysis` module.


Built-in analysis methods
------------------------------
.. currentmodule:: ramanspy.analysis

`RamanSPy` provides many of the commonly-used techniques for spectral analysis. This includes a broad collection of
methods for decomposition, clustering and spectral unmixing, which, similarly to preprocessing methods, can be directly
interfaced through their ``apply()`` method.

Decomposition
^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/analysis/decompose/

    decompose.PCA
    decompose.NMF
    decompose.ICA

.. seealso:: Check the :ref:`Built-in decomposition methods` tutorial for more information about how to access and use
             the decomposition algorithms built into `RamanSPy`.


Clustering
^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/analysis/cluster/

    cluster.KMeans

.. seealso:: Check the :ref:`Built-in clustering methods` tutorial for more information about how to access and use the
             clustering algorithms built into `RamanSPy`.


Spectral unmixing
^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/analysis/unmix/

    unmix.PPI
    unmix.FIPPI
    unmix.NFINDR
    unmix.VCA

.. seealso:: Check the :ref:`Built-in unmixing methods` tutorial for more information about how to access and use the
             spectral unmixing algorithms built into `RamanSPy`.



Integrative analysis
-----------------------
Because of `RamanSPy's` data management design, data stored within the package can easily be integrated into the rest of
the Python analysis ecosystem, including most frameworks for statistical and machine learning modelling. As such methods
are increasingly often utilised for Raman spectroscopic research, we believe this will be a feature of paramount
importance for future research in the area.

.. seealso:: Check the :ref:`Integrative analysis: Support Vector Machine (SVM) classification` and :ref:`Integrative analysis: Neural Network (NN) classification`
             tutorials for more information about how to integrate `RamanSPy` with other Python analysis packages.
