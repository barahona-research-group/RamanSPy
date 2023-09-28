"""
Using built-in metrics
--------------------------------------

In this tutorial, we will see how to access and use the built-in metrics available in `RamanSPy`.
"""
import numpy as np

import ramanspy

# %%
# To access the built-in metrics, simply use the :mod:`ramanspy.metrics` module.

# %%
# Before we make use of the metrics, let us first define some dummy data to work with. We will create two spectra, one
# that is a sine wave and another that is a cosine wave. We will then use the metrics to compare the two spectra.
spectral_axis = np.linspace(100, 3600, 1500)

sine_data = np.sin(spectral_axis/120)
cosine_data = np.cos(spectral_axis/120)

sine_spectrum = ramanspy.Spectrum(sine_data, spectral_axis)
cosine_spectrum = ramanspy.Spectrum(cosine_data, spectral_axis)

# %%
# We can visualise the two spectra with the data visualisation tools in `RamanSPy`.
_ = ramanspy.plot.spectra([sine_spectrum, cosine_spectrum], plot_type="single", label=["Sine", "Cosine"])

# %%
# We can then measure the distance or similarity of the two spectra using the built-in metrics. For instance, we can
# use the :meth:`ramanspy.metrics.MAE` to measure the mean absolute error between the two spectra.
ramanspy.metrics.MAE(sine_spectrum, cosine_spectrum)

# %%
# To double-check that the metric is working as expected, we can also measure the distance between the seme spectrum.
ramanspy.metrics.MAE(sine_spectrum, sine_spectrum)

# %%
# Other metrics that are available include the :meth:`~ramanspy.metrics.MSE`, :meth:`~ramanspy.metrics.RMSE`, :meth:`~ramanspy.metrics.SAD`, :meth:`~ramanspy.metrics.SID`.
# For more information about these metrics, refer to their documentation.

# %%
# For instance, we can use the :meth:`~ramanspy.metrics.SAD` to measure the spectral angle distance between the two spectra.
ramanspy.metrics.SAD(sine_spectrum, cosine_spectrum)
