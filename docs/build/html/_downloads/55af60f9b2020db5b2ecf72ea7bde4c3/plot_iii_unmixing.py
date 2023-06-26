"""
Built-in unmixing methods
================================

In this tutorial, we will use the `RamanSPy's` built-in methods for spectral unmixing to perform N-FINDR and Fully-Constrained
Least Squares (FCLS) on a Raman spectroscopic image. To do that, we will employ `RamanSPy` to analyse the fourth layer of
the volumetric :ref:`Volumetric cell data` provided in `RamanSPy`.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 5
# sphinx_gallery_end_ignore

import ramanspy

dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

cell_layer = volumes[0].layer(5)  # only selecting the fourth layer of the volume


# %%
# Let's first plot a spectral slice across the 1008 cm :sup:`-1` band of the image to visualise what has been captured in the image.
cell_layer.plot(bands=[1008])


# %%
# We can also visualise a specific spectrum within the image.
cell_layer[30, 30].plot()


# %%
# We may need to first preprocess the spectral image to improve the results of our consecutive analysis.
preprocessing_pipeline = ramanspy.preprocessing.Pipeline([
    ramanspy.preprocessing.misc.Cropper(region=(500, 1800)),
    ramanspy.preprocessing.despike.WhitakerHayes(),
    ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    ramanspy.preprocessing.baseline.ASLS(),
    ramanspy.preprocessing.normalise.MinMax(pixelwise=False),
])
preprocessed_cell_layer = preprocessing_pipeline.apply(cell_layer)


# %%
# To check the effect of our preprocessing protocol, we can re-plot the same spectral slice as before
preprocessed_cell_layer.plot(bands=[1008])


# %%
# as well as the same spectra we visualised before.
preprocessed_cell_layer[30, 30].plot()


# %%
# Then, we can use `RamanSPy` to perform N-FINDR with 4 endmembers, followed by FCLS.
nfindr = ramanspy.analysis.unmix.NFINDR(n_endmembers=4, abundance_method='fcls')
# %%
#
abundance_maps, endmembers = nfindr.apply(preprocessed_cell_layer)


# %%
# As a last step, we can use `RamanSPy's` :meth:`ramanspy.plot.spectra` and :meth:`ramanspy.plot.image` methods to visualise the
# calculated endmember signatures and the corresponding fractional abundance maps.
ramanspy.plot.spectra(endmembers, preprocessed_cell_layer.spectral_axis, plot_type="single stacked", label=[f"Endmember {i + 1}" for i in range(len(endmembers))])

# %%
#
ramanspy.plot.image(abundance_maps, title=[f"Component {i + 1}" for i in range(len(abundance_maps))])
