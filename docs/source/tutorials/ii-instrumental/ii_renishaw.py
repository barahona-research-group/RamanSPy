"""
Loading Renishaw data
--------------------------------------

Users can use `RamanSPy` to load .wdf files as exported from `Renishaw's WiRE <https://www.renishaw.com/en/raman-software--9450>`_ software.
This can be done through the :meth:`ramanspy.load.renishaw` method by simply providing the name of the file of interest.
"""
import ramanspy

# %%
# The method itself will parse the data to the correct spectral data container, which you can then use as usual.
raman_data = ramanspy.load.renishaw("path/to/file/renishaw_data.wdf")
