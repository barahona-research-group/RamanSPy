=================================
Data loading
=================================
.. currentmodule:: ramanspy.load

`RamanSPy` can be used to easily load experimental Raman spectroscopic data from different instruments and manufacturers.
This is enabled through the introduction of custom data loading functions, which can parse and load different data formats
into the appropriate :ref:`data containers`.

Loading methods are available within the :mod:`ramanspy.load` module.


WITec Suite (WITec)
=========================
.. autofunction:: witec


.. seealso:: Check the :ref:`Loading WITec data` tutorial for more information about how to load data from WITec instruments.


WiRE (Renishaw)
=========================
.. autofunction:: renishaw


.. seealso:: Check the :ref:`Loading Renishaw data` tutorial for more information about how to load data from Renishaw instruments.


OceanView (Ocean Insight)
=========================
.. autofunction:: ocean_insight


.. seealso:: Check the :ref:`Loading Ocean Insight data` tutorial for more information about how to load data from Ocean Insight instruments.


LabSpec (HORIBA)
=========================
.. autofunction:: labspec


Other
========================
.. currentmodule:: ramanspy

To allow the loading of other data file formats, simply define an appropriate data loading function, which parses the
specific spectroscopic files of interest into :class:`SpectralContainer` objects (e.g. :class:`SpectralContainer`, :class:`Spectrum`,
:class:`SpectralImage` or :class:`SpectralVolume`).

Then, the core of `RamanSPy` will automatically deal with the rest of the workflow for you and you will be able to access
all its preprocessing, analysis and data visualisation functionalities as usual.


.. seealso:: Check the :ref:`Loading other data` tutorial for more information about how to use `RamanSPy` to load other types of data.
