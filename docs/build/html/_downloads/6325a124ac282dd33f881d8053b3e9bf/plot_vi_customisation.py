"""
Customising plots
====================================

`RamanSPy's` plotting methods are built on top of `matplotlib <https://matplotlib.org>`_ and so inherit most of `matplotlib's`
customisability.

Below, we will highlight some of that customisability.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -2
# sphinx_gallery_end_ignore

from matplotlib import pyplot as plt

import ramanspy

# %%
# As an example, we will use the :ref:`Volumetric cell data` provided within `RamanSPy`.
dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

# %%
# We will use the first volume of the dataset, which is a 3D image of a cell.
cell_volume = volumes[0]

# crop the data
cropper = ramanspy.preprocessing.misc.Cropper(region=(300, None))
cell_volume = cropper.apply(cell_volume)

# get the fourth layer of the volume as an example spectral image
cell_layer = cell_volume.layer(4)

# define example volume and image slices and spectra
cell_volume_slices = [cell_volume.band(1600), cell_volume.band(2930), cell_volume.band(3300)]
cell_layer_slices = [cell_layer.band(1600), cell_layer.band(2930), cell_layer.band(3300)]
spectra = [cell_layer[20, 30], cell_layer[30, 20], cell_layer[10, 20]]

# %%
# Default behaviour
# -------------------
# By default, `RamanSPy` inherits `matplotlib's` default `settings <https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html>`_.
#
# This looks as follows:

# %%

# spectra plots
ramanspy.plot.spectra(spectra)

# %%

# image plots
ramanspy.plot.image(cell_layer_slices)

# %%

# volume plots
ramanspy.plot.volume(cell_volume_slices)

# %%
# Parameter control
# -------------------
# Users can control plot characteristics by changing the parameters of the plotting methods in `RamanSPy`. As these extend
# `matplotlib` methods, we can control them just as we control the underlying `matplotlib` methods.
#
# For instance:

# %%

# changing the color of spectra plots
ramanspy.plot.spectra(spectra, color='red')

# %%

# changing the color and type of spectra plots
ramanspy.plot.spectra(spectra, color=['blue', 'green', 'red'], linestyle='-.')

# %%

# changing the color of image plots
ramanspy.plot.image(cell_layer_slices, color=['blue', 'green', 'red'])

# %%

# changing the color of volume plots
ramanspy.plot.volume(cell_volume_slices, color=['blue', 'green', 'red'])


# %%
# .. seealso:: For more information about the available parameters you can change and how to do that, check the documentation
#              of the :meth:`ramanspy.plot.spectra`, :meth:`ramanspy.plot.image` and :meth:`ramanspy.plot.volume` methods.


# %%
# Settings control
# -------------------
# Users can also change the behaviour of `RamanSPy's` visualisation tools by directly `changing matplotlib's settings <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html>`_.
#
# For instance:

# %%
# Changing the color palette
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# changing the colormap to 'jet'
plt.rc('image', cmap='jet')


# spectra plots
ramanspy.plot.spectra(spectra)


# image plots
ramanspy.plot.image(cell_layer_slices)


# volume plots
ramanspy.plot.volume(cell_volume_slices)

# %%
# Changing other settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# changing the size of the plot
plt.rc('figure', figsize=(12, 3))

ramanspy.plot.spectra(spectra[0])

# %%

# changing the width of line plots and making them dashed
plt.rc('lines', linewidth=4, linestyle='-.')

ramanspy.plot.spectra(spectra[0])

# %%
# .. seealso:: For more information about the available settings you can change and how to do that, check `matplotlib's settings <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html>`_.
