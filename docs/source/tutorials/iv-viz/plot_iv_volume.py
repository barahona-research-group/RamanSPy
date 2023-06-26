"""
Visualising volumetric data
=================================

`RamanSPy` aids the visualisation of volumetric Raman spectroscopic data. This can be done by using the
:meth:`ramanspy.plot.volume` or :meth:`ramanspy.SpectralVolume.plot` methods.
"""

# %%
# In this example, we will showcase that using the :ref:`Volumetric cell data` provided in `RamanSPy`.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_end_ignore

import ramanspy

dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

# %%
# We will use the first volume of the dataset, which is a 3D image of a cell.
cell_volume = volumes[0]

# %%
# Visualising volumetric data follows the same workflow as plotting imaging Raman data (check the :ref:`Visualising imaging data` tutorial).
# Once we have a :class:`ramanspy.SpectralVolume` object, we aan simply invoke its :meth:`ramanspy.SpectralVolume.plot` method
# to plot a spectral slice across a specific band.
cell_volume.plot(bands=[1008])

# %%
# Again, we can alternatively use the :meth:`ramanspy.plot.volume` method.
ramanspy.plot.volume(cell_volume.band(1008))

# %%
# Similarly to when plotting imaging data, we can also change various parameters to make the plot more informative and precise.
ramanspy.plot.volume(cell_volume.band(1008), title="Cell imaged with Raman spectroscopy", cbar_label=f"Peak intensity at 1008cm$^{{{-1}}}$")


# %%
# We can also save the plot to a file here, as well.
ax = ramanspy.plot.volume(cell_volume.band(1008))
ax.figure.savefig("v.png")

# %%
# Check the rest of the documentations of the two functions for more information of the available parameters.
