"""
Visualising imaging data
=============================

`RamanSPy` allows the visualisation of Raman imaging data. Visualising imaging data can be achieved by utilising the
:meth:`ramanspy.plot.image` or :meth:`ramanspy.SpectralImage.plot` method.
"""

# %%
# To show how that can be done, in this example, we will use the :ref:`Volumetric cell data` provided in `RamanSPy`. Below, we load
# the volumetric data across a cell and select a particular layer from the volume (here, this is the fourth layer).

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_end_ignore

import ramanspy
dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

cell_layer = volumes[0].layer(4)  # only selecting the fourth layer of the volume

# %%
# Having loaded a :class:`ramanspy.SpectralImage` object, we can directly plot a spectral slice across a specific band of
# the image using its :meth:`ramanspy.plot.image` method. Here, this is the 1008 cm :sup: `-1` band.
ramanspy.plot.image(cell_layer.band(1008))

# %%
# We can achieve the same behaviour using the :meth:`ramanspy.SpectralImage.plot` method, too.
cell_layer.plot(bands=[1008])


# %%
# As usual, `RamanSPy` provides a broad control over the characteristics of the plot. For instance, we can add more informative
# title, axis labels, colorbar label, colour schemes, etc.
ramanspy.plot.image(cell_layer.band(1008), title="Cell imaged with Raman spectroscopy", cbar_label=f"Peak intensity at 1008cm$^{{{-1}}}$")


# %%
# Users can also use `RamanSPy` to save the image to a file.
ax = ramanspy.plot.image(cell_layer.band(1008))
ax.figure.savefig("cell_image.png")


# %%
# It is also possible to plot several images at the same time. When doing that, separate plots will be produced.
cell_layer.plot(bands=[1600, 1008])

# or ramanspy.plot.image([cell_layer.band(1600), cell_layer.band(1008)])


# %%
# Check the rest of the documentations of the two functions for more information of the available parameters.
