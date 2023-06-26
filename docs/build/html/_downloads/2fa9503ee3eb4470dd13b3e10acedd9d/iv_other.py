"""
Loading other data
--------------------------------------

Users can use `RamanSPy` to load other data files, too. To do so, one simply has to parse the file they are interested in
to the correct spectral data container. Then, it can be directly integrated into the rest of the package.
"""
import ramanspy


# %%
# For instance, if we are interested in loading single spectra from two-column .csv files containing the Raman
# wavenumber axis (in a column called "Wavenumber") and the corresponding intensity values (in a column called "Intensity")
# respectively. Then, we can define a function, which parses such files as follows:

import pandas as pd

def parsing_csv(csv_filename):
    data = pd.read_csv(csv_filename)

    # parse and load data into spectral objects
    spectral_data = data["Wavenumber"]
    spectral_axis = data["Intensity"]

    raman_spectrum = ramanspy.Spectrum(spectral_data, spectral_axis)

    return raman_spectrum

# %%
# Then, we can use the package to load data from such files into `RamanSPy` and use the package to analyse the data.
raman_spectrum = parsing_csv("path/to/file/spectrum.csv")
