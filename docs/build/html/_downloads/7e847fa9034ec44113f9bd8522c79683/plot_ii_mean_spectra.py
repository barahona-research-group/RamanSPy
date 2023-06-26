"""
Visualising spectral distributions
===================================

Sometimes, we have plenty of spectra we wish to visualise. To appropriately do that, it is more appropriate to visualise
summary statistics of the (groups of) spectra we want to investigate, such as the mean of the corresponding
collection of spectra and/or describe its spectral distribution. This is why `raman` offers a wide variety
of visualisation tools for plotting distributions of Raman spectra, which can be accessed via the :meth:`raman.plot.mean_spectra` method.

.. note:: The behaviour of the :meth:`raman.plot.mean_spectra` method closely follows that of the :meth:`raman.plot.spectra`
method. Hence, readers are advised to first check its documentation, as well as the :ref:`Visualising spectra` tutorial.
"""

# %%
# To show that, we will this time use the training dataset of the :ref:`Bacteria data` provided within `raman`.
# For this tutorial, we will only use the data for the first 10 bacteria species and use only 10 spectra per species.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore

import ramanspy

dir_ = r"../../../../src/ramanspy\data\bacteria_data"
X_train, y_train = ramanspy.datasets.bacteria("train", folder=dir_)

# %%
bacteria_lists = [[X_train[i:i+10, :]] for i in range(0, X_train.shape[0], 2000)]

bacteria_sample = bacteria_lists[:10]
bacteria_sample_labels = [f"Species {int(y_train[i*2000])}" for i in range(0, 10)]


# %%
# Single plots
# -------------------
# As with single spectra, we can also visualise a single group of spectra as a distribution.

# %%
# Even for a small number of spectra (e.g. 3-5), it becomes hard to visualise them in a single plot. So, instead, we
# can use `raman` to only highlight the mean of a group and a confidence interval around it (a 95% CI based on normal distribution).
# This can be done by setting the ``dist`` parameter of the :meth:`raman.plot.mean_spectra` method to ``True`` (default behaviour).
ramanspy.plot.mean_spectra(bacteria_sample[0], plot_type='single')

# %%
# To plot more groups in a single plot, just provide the group list as follows:
ramanspy.plot.mean_spectra(bacteria_sample, plot_type='single')


# %%
# If we prefer, we can plot the individual spectra within the group instead of the CI by setting the ``dist`` to ``False``.
#
# Note that this method is not preferred when we have a large number of spectra within the group(s) we are interested in.
ramanspy.plot.mean_spectra(bacteria_sample[0], plot_type='single', dist=False)



# %%
# Separate plots
# -------------------
# To improve the readability of the plot, we can also visualise distributions in separate plots:

ramanspy.plot.mean_spectra(bacteria_sample[:3], plot_type='separate')


# %%
# Stacked plots
# -------------------
# But that is still not ideal if we want to compare the distributions. In such cases, it is more informative to use `stacked` plots.
ramanspy.plot.mean_spectra(bacteria_sample, plot_type='stacked')


# %%
# Single stacked plots
# ---------------------
# Or `single stacked plots`.
ramanspy.plot.mean_spectra(bacteria_sample, plot_type="single stacked")


# %%
# We can also add more informative title, legend, axis labels, etc.
ramanspy.plot.mean_spectra(bacteria_sample, plot_type="single stacked", label=bacteria_sample_labels, title='Bacteria identification using Raman spectroscopy')
