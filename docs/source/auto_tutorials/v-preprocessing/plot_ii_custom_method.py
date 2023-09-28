"""
Custom methods
--------------------------------------

Users can use `RamanSPy` to also define their own preprocessing methods, which can then be directly integrated into
the preprocessing core of `RamanSPy`. This can be done by wrapping custom preprocessing methods into :class:`ramanspy.preprocessing.PreprocessingStep` instances.

Below, we will use `RamanSPy` to define and apply a custom preprocessing method to the volumetric :ref:`Volumetric cell data` provided in `RamanSPy`.

"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_end_ignore

import ramanspy

dir_ = r'../../../../data/kallepitis_data'

volumes = ramanspy.datasets.volumetric_cells(cell_type='THP-1', folder=dir_)

# %%
# We will use the first volume
cell_volume = volumes[0]

# selecting a random spectrum for visualisation purposes
random_spectrum = cell_volume[25, 25, 5]
random_spectrum.plot(title='Original Raman spectra')


# %%
# To do so, users can define their own preprocessing methods, which must be of the form given below and return the updated intensity_data and spectral_axis.
def func(intensity_data, spectral_axis, **kwargs):
    # Preprocess intensity_data and spectral axis
    updated_intensity_data = ...
    updated_spectral_axis = ...

    return updated_intensity_data, updated_spectral_axis


# %%
# For instance, we can define a simple example function, which simply subtracts a given offset from each value in the
# intensity_data array as follows:
def offset_func(intensity_data, spectral_axis, *, offset):
    return intensity_data - offset, spectral_axis


# %%
# Then, one must simply wrap the function of interest using the :class:`ramanspy.preprocessing.PreprocessingStep` class.
# That is done by creating a :class:`ramanspy.preprocessing.PreprocessingStep` object by invoking its
# :meth:`ramanspy.preprocessing.PreprocessingStep.__init__` method on the function one wants to wrap and the *args, **kwargs needed.
offsetter = ramanspy.preprocessing.PreprocessingStep(offset_func, offset=500)


# %%
# Having done that, the :class:`ramanspy.preprocessing.PreprocessingStep` object can now be used as any of `RamanSPy's`
# preprocessing methods through its :meth:`ramanspy.preprocessing.PreprocessingStep.apply` method.
preprocessed_random_spectrum = offsetter.apply(random_spectrum)
preprocessed_random_spectrum.plot()

# %%
# The custom method can also be directly integrated into pipelines as follows:
preprocessing_pipeline = ramanspy.preprocessing.Pipeline([
    ramanspy.preprocessing.despike.WhitakerHayes(),
    ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    offsetter
])
preprocessed_random_spectrum = preprocessing_pipeline.apply(random_spectrum)

# %%
# Visualising the preprocessed spectrum.
preprocessed_random_spectrum.plot(title='Preprocessed Raman spectra')

# %%
# .. note:: Custom preprocessing methods defined with `RamanSPy` work equally well on the other spectral data containers, as well as on collection of those.
