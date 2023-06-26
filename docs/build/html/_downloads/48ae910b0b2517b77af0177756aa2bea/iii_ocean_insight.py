"""
Loading Ocean Insight data
--------------------------------------

Users can use `RamanSPy` to load single spectra .txt files acquired using Ocean Insight Raman instruments (version xxx).

This can be done through the :meth:`ramanspy.load.ocean_insight` method by again simply providing the name of the file of interest.
"""
import ramanspy

# %%
# You can use the method to load single spectra.
raman_spectrum = ramanspy.load.ocean_insight("path/to/file/ocean_insight_spectrum.txt")
