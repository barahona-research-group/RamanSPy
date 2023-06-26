"""
Loading the Bacteria dataset
--------------------------------------

In this tutorial, we will see how to load the :ref:`Bacteria data` available in `RamanSPy`.
"""
import matplotlib.pyplot as plt
import numpy as np

import ramanspy

# %%
# To load a specific dataset split of the data, simply use the :meth:`ramanspy.datasets.bacteria` method and indicate
# the split you want to load and the directory where the corresponding dataset has been downloaded to. For instance:
dir_ = r"../../../../data/bacteria_data"

X_train, y_train = ramanspy.datasets.bacteria("val", folder=dir_)

# %%
# Loading the labels:
y_labels, _ = ramanspy.datasets.bacteria("labels")

# %%
# Organising the spectra by species:
spectra = [[X_train[y_train == species_id]] for species_id in list(np.unique(y_train))]

# %%
# Normalise the spectra using min-max normalisation.
spectra_ = ramanspy.preprocessing.normalise.MinMax().apply(spectra)

# %%
# Plot the mean spectra of each species.
plt.figure(figsize=(6.5, 9))
ramanspy.plot.mean_spectra(spectra_, label=y_labels, plot_type="single stacked", title=None)


# %%
# For more information about the :meth:`~ramanspy.datasets.bacteria` method, refer to its documentation:
help(ramanspy.datasets.bacteria)
