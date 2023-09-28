"""
Built-in decomposition methods
================================

In this example, we will use `RamanSPy` to perform Principal Component Analysis (PCA) to decompose a Raman spectroscopic
image into its constituent components.

To do that, we will use the volumetric :ref:`Volumetric cell data` available in `RamanSPy`. In particular,
we will decompose the fourth layer of the provided volumetric dataset.
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
# We will then perform PCA with 4 components using `RamanSPy`.
pca = ramanspy.analysis.decompose.PCA(n_components=4)
# %%
#
projections, components = pca.apply(preprocessed_cell_layer)


# %%
# Having derived the PCA components and the corresponding projections, we can use `RamanSPy's` :meth:`ramanspy.plot.spectra`
# and :meth:`ramanspy.plot.image` methods to visualise them.
ramanspy.plot.spectra(components, preprocessed_cell_layer.spectral_axis, plot_type="single stacked", label=[f"Component {i + 1}" for i in range(len(components))])

# %%
#
ramanspy.plot.image(projections, title=[f"Projection {i + 1}" for i in range(len(projections))])
