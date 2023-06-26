=================================
Data containers
=================================
.. currentmodule:: ramanspy

Experimental Raman spectroscopic data can be very diverse, depending on their data acquisition modality
and instrumental origin. This usually makes any consecutive operations heavily application-specific, which in turn
impedes the development of transferable and reusable workflows.

`RamanSPy` resolves this by decoupling its data management core from its preprocessing and analysis functionalities.
This is achieved by establishing efficient and scalable data representation classes built upon `numpy ndarrays <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_,
which capture and define relevant information and behaviour in the background to allow a smooth, modification-free experience,
regardless of the data of interest.

This includes a generic spectral container class :class:`SpectralContainer`, as well as
the more specialised :class:`Spectrum`, :class:`SpectralImage` and :class:`SpectralVolume` classes,
which correspond to single spectra, Raman imaging data and volumetric Raman data respectively.

For the most part, the construction of these classes is automated through the various :ref:`data loading` and :ref:`datasets`
methods in `RamanSPy`. However, the containers can be populated manually as well, if required.

Generic container
========================
.. autoclass:: SpectralContainer
    :members:
    :undoc-members:


.. seealso:: Check the :ref:`Storing generic data` tutorial for more information about how to define and use spectral data containers in `RamanSPy`.


Specialised containers
========================

Additional data-specific information is added through the specialised :class:`Spectrum`, :class:`SpectralImage` and
:class:`SpectralVolume` classes. These ensure that the data management of single spectra, imaging and volumetric Raman data
is appropriately dealt with without the need for any data-related user modifications.


Spectra
--------------------------
.. autoclass:: Spectrum
    :members:
    :undoc-members:
    :show-inheritance:


.. seealso:: Check the :ref:`Storing spectra` tutorial for more information about how to define and use :ref:`Spectrum` data containers in `RamanSPy`.


Imaging
----------------------
.. autoclass:: SpectralImage
    :members:
    :undoc-members:
    :show-inheritance:


.. seealso:: Check the :ref:`Storing imaging data` tutorial for more information about how to define and use :ref:`SpectralImage` data containers in `RamanSPy`.


Volumetric
----------------------------
.. autoclass:: SpectralVolume
    :members:
    :undoc-members:
    :show-inheritance:


.. seealso:: Check the :ref:`Storing volumetric data` tutorial for more information about how to define and use :ref:`SpectralVolume` data containers in `RamanSPy`.
