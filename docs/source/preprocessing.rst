Preprocessing
********************
.. currentmodule:: ramanspy.preprocessing

`RamanSPy` offers extensive preprocessing support that alleviates the common burdens of spectral preprocessing and enables
the construction and execution of complex preprocessing procedures with minimal software requirements.

Users can access and use the variety of preprocessing techniques and protocols built into `RamanSPy`, as well as use the
package to define custom preprocessing algorithms and pipelines.

All of the preprocessing methods and pipelines are implemented as :class:`PreprocessingStep` instances, which standardises
their behaviour and allows for their easy integration into various preprocessing pipelines.

These have been designed to be as flexible and data-agnostic as possible, so that they can be applied to any type of
Raman spectroscopic data loaded into the framework.

Preprocessing support is provided by the :mod:`ramanspy.preprocessing` module.

Algorithms
=========================
The behaviour of preprocessing procedures is defined by the :class:`PreprocessingStep` class.

.. autoclass:: PreprocessingStep
    :members:
    :undoc-members:


Built-in preprocessing methods
-----------------------------------
`RamanSPy` provides many of the commonly-used techniques for spectral preprocessing. This includes a broad collection of
methods for cosmic ray removal, denoising, baseline correction, normalisation and other preprocessing procedures.

These are built into `RamanSPy` as classes extending :class:`PreprocessingStep` and can thus be readily used via their
``apply()`` method. The built-in methods include:

Miscellaneous
^^^^^^^^^^^^^^^^^^^
.. autosummary::

    misc.Cropper
    misc.BackgroundSubtractor

Cosmic rays removal
^^^^^^^^^^^^^^^^^^^^
.. autosummary::

    despike.WhitakerHayes

Denoising
^^^^^^^^^^^^^^^^^^^
.. autosummary::

    denoise.SavGol
    denoise.Whittaker
    denoise.Kernel
    denoise.Gaussian

Baseline correction
^^^^^^^^^^^^^^^^^^^

Least squares
"""""""""""""""""""""""""
.. autosummary::

    baseline.ASLS
    baseline.IASLS
    baseline.AIRPLS
    baseline.ARPLS
    baseline.DRPLS
    baseline.IARPLS
    baseline.ASPLS


Polynomial fitting
""""""""""""""""""""""""""""

.. autosummary::

    baseline.Poly
    baseline.ModPoly
    baseline.Poly
    baseline.ModPoly


Other
""""""

.. autosummary::

    baseline.Goldindec
    baseline.IRSQR
    baseline.CornerCutting
    baseline.FABC


Normalisation/Scaling
^^^^^^^^^^^^^^^^^^^
.. autosummary::

    normalise.Vector
    normalise.MinMax
    normalise.MaxIntensity
    normalise.AUC


.. seealso:: Check the :ref:`Built-in methods` tutorial for more information about how to access and use the preprocessing algorithms built into `RamanSPy`.


Custom algorithms
-------------------
Alternatively, users can use `RamanSPy` to create their own preprocessing methods by wrapping preprocessing functions of the
correct type within :class:`PreprocessingStep` instances.

.. seealso:: Check the :ref:`Custom methods` tutorial for more information about how to define custom preprocessing methods using `RamanSPy`.


Pipelines
========================
In most applications, there are several preprocessing procedures, which need to be performed on the experimental Raman
spectroscopic data one's working with before proceeding with the consecutive analysis. The construction and
customisation of such complex preprocessing pipelines are usually software-intensive tasks, which are
unnecessarily challenging.

This is why `RamanSPy` also provides tools for the smooth development of preprocessing pipelines. This is made
possible through the introduction of the :class:`Pipeline` class.

.. autoclass:: Pipeline
    :members:
    :undoc-members:


Custom pipelines
-------------------
The preprocessing methods offered by the package and other custom algorithms (wrapped as :class:`PreprocessingStep` instances),
can easily be stacked together using `RamanSPy` into complete multi-layered preprocessing pipelines that work just as single
:class:`PreprocessingStep` instances do.

To create a preprocessing pipeline, simply wrap a list of preprocessing methods within a :class:`Pipeline` instance.

.. seealso:: Check the :ref:`Custom pipelines` tutorial for more information about how to construct and execute custom preprocessing pipelines.


Established protocols
----------------------
`RamanSPy` also provides preprocessing protocols already proposed in the literature. This allows users
to select pre-configured preprocessing pipelines without having to worry about the choice of methods and parameters.
These can be accessed through the following methods:

.. autosummary::

    protocols.default
    protocols.default_fingerprint
    protocols.articular_cartilage

.. seealso:: Check the :ref:`Built-in protocols` tutorial for more information about how to access and use the preprocessing protocols built into `RamanSPy`.
