Overview
=================================

Code example
-------------
Below is a simple example of how `RamanSPy` can be used to load, preprocess and analyse Raman spectroscopic data. Here,
we load a data file from a commercial Raman instrument; apply a preprocessing pipeline consisting of spectral cropping,
cosmic ray removal, denoising, baseline correction and normalisation; perform spectral unmixing; and visualise the results.

For more examples, check out :ref:`Tutorials` and :ref:`Examples`.

.. code::

    import ramanspy as rp

    # load data
    image_data = rp.load.witec("<PATH>")

    # apply a preprocessing pipeline
    pipeline = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=(700, 1800)),
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
        rp.preprocessing.baseline.ASPLS(),
        rp.preprocessing.normalise.MinMax()
    ])
    data = pipeline.apply(image_data)

    # perform spectral unmixing
    nfindr = rp.analysis.unmix.NFINDR(n_endmembers=5)
    amaps, endmembers = nfindr.apply(data)

    # plot results
    rp.plot.spectra(endmembers)
    rp.plot.image(amaps)
    rp.plot.show()

|

Features
------------
`RamanSPy` offers a range of features to support Raman research and analysis. This includes the following:

Complete workflow support
""""""""""""""""""""""""""
`RamanSPy` streamlines the entire Raman spectroscopic data analysis lifecycle by providing accessible, easy-to-use tools for
loading, preprocessing, analysing and visualising diverse Raman spectroscopic data. All functionalities of
`RamanSPy` are completely application-agnostic and work equally well on any data loaded into the framework, regardless of
their spectroscopic modality (single-point spectra, imaging, volumetric) and instrumental origin. This allows users
to construct entire analysis workflows with minimal software requirements which are out-of-the-box transferable across
different datasets and projects.

Preprocessing pipelines
""""""""""""""""""""""""""""""""""
The package also improves the availability, consistency and reproducibility of preprocessing processes by
providing a pipelining infrastructure that streamlines the compilation, customisation and execution of complex preprocessing
pipelines, as well as provides a library of already established and validated preprocessing protocols. This not only improves
preprocessing automation, but also ensures that the same preprocessing pipeline is applied consistently across different,
as well as that users can easily access and reproduce the preprocessing steps used in published research.

Integrative analysis
""""""""""""""""""""""""
Furthermore, `RamanSPy` has been designed such that it offers direct integration with the entire Python ecosystem, thereby
allowing smooth incorporation with other Python packages for spectroscopic research (e.g. `pysptools <https://pysptools.sourceforge.io/>`_), statistical modelling
and machine learning (e.g. `scikit-learn <https://scikit-learn.org/>`_), deep learning (e.g. `pytorch <https://pytorch.org/>`_, `tensorflow <https://www.tensorflow.org/>`_) and many others.
With that, we aim to facilitate the integration of new (AI-based) methods and applications into the Raman spectroscopic workflow
and catalyse the emerging effort to bridge the gap between Raman spectroscopy and AI & ML.

Model development
""""""""""""""""""""""""
Finally, `RamanSPy` has been equipped with a range of tools for algorithmic development and evaluation, including a library
of diverse Raman spectroscopic datasets, as well as a set of metrics for the evaluation of model performance. This allows users
to more efficiently and consistently develop and evaluate new (AI-based) methods and algorithms for Raman spectroscopy applications.
