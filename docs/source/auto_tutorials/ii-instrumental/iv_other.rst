
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_tutorials/ii-instrumental/iv_other.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_tutorials_ii-instrumental_iv_other.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_tutorials_ii-instrumental_iv_other.py:


Loading other data
--------------------------------------

Users can use `RamanSPy` to load other data files, too. To do so, one simply has to parse the file they are interested in
to the correct spectral data container. Then, it can be directly integrated into the rest of the package.

.. GENERATED FROM PYTHON SOURCE LINES 8-11

.. code-block:: default

    import ramanspy



.. GENERATED FROM PYTHON SOURCE LINES 12-15

For instance, if we are interested in loading single spectra from two-column .csv files containing the Raman
wavenumber axis (in a column called "Wavenumber") and the corresponding intensity values (in a column called "Intensity")
respectively. Then, we can define a function, which parses such files as follows:

.. GENERATED FROM PYTHON SOURCE LINES 15-29

.. code-block:: default


    import pandas as pd

    def parsing_csv(csv_filename):
        data = pd.read_csv(csv_filename)

        # parse and load data into spectral objects
        spectral_data = data["Wavenumber"]
        spectral_axis = data["Intensity"]

        raman_spectrum = ramanspy.Spectrum(spectral_data, spectral_axis)

        return raman_spectrum


.. GENERATED FROM PYTHON SOURCE LINES 30-31

Then, we can use the package to load data from such files into `RamanSPy` and use the package to analyse the data.

.. GENERATED FROM PYTHON SOURCE LINES 31-32

.. code-block:: default

    raman_spectrum = parsing_csv("path/to/file/spectrum.csv")


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.000 seconds)


.. _sphx_glr_download_auto_tutorials_ii-instrumental_iv_other.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example




    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: iv_other.py <iv_other.py>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: iv_other.ipynb <iv_other.ipynb>`
