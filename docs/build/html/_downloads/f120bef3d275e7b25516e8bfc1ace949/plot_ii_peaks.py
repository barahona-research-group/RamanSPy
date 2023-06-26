"""
Visualising peaks
======================

We can visualise the peaks in a spectrum by using the :meth:`ramanspy.plot.peaks` method.

"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore

import ramanspy

dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

# %%
# We will use the first volume of the dataset, which is a 3D image of a cell.
cell_volume = volumes[0]

# %%
# We will use the sixth layer (given by index 5) of the volume as an example spectral image.
cell_layer = cell_volume.layer(5)

# %%
# We will select a specific spectra from the image.
selected_spectrum = cell_layer[20, 30]

# %%
# We will first preprocess the spectral spectrum
preprocessing_pipeline = ramanspy.preprocessing.Pipeline([
    ramanspy.preprocessing.misc.Cropper(region=(500, 1800)),
    ramanspy.preprocessing.despike.WhitakerHayes(),
    ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    ramanspy.preprocessing.baseline.ASLS(),
    ramanspy.preprocessing.normalise.MinMax(pixelwise=False),
])
preprocessed_spectrum = preprocessing_pipeline.apply(selected_spectrum)

# %%
# We can now visualise the peaks in the spectrum.
_ = ramanspy.plot.peaks(preprocessed_spectrum, prominence=0.15)
