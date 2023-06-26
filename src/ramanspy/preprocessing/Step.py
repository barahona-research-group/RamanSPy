from typing import Union, Callable
import copy
from typing import final, List

from ..core import SpectralObject


class PreprocessingStep:
    """
    A class that defines preprocessing logic.

    Encapsulate preprocessing methods that transform the intensity values and spectral axis of Raman data.

    To define a preprocessing procedure that can be applied to any Raman spectroscopic data, you must wrap a predefined
    preprocessing method using this class, which in turn streamlines any consecutive operations.

    Parameters
    ----------
    method : Callable
        A Callable object (e.g. a method) which defines how the preprocessing step alters spectral objects. Its ``__call__`` method
        must have signature of the form: ``__call__(intensity_data, spectral_axis, *args, **kwargs)``, where ``intensity_data``
        is an ndarray of arbitrary shape defining the intensity values to process, whose last axis is the spectral axis,
        ``spectral_axis`` - a 1D ndarray defining the spectral axis to process (in Raman wavenumber cm :sup:`-1` units),
        ``*args`` - other positional arguments, and ``**kwargs`` - other keyword arguments.
    **kwargs :
        Any keyword arguments the Callable needs in its ``__call__`` method.


    .. note:: One has to use the :class:`PreprocessingStep` class only when devising and integrating custom preprocessing methods (check :ref:`Custom algorithms`).

              All preprocessing methods built into `RamanSPy` can be directly accessed and used as indicated in :ref:`Built-in preprocessing methods`.

    Example
    ----------
    
    .. code:: 
    
        from ramanspy import preprocessing
       
        # Defining some preprocessing function of the correct type
        def preprocessing_func(intensity_data, spectral_axis, **kwargs):
            # Preprocess intensity_data and spectral axis
            ...

            return updated_intensity_data, updated_spectral_axis
       
        # wrapping the function into a PreprocessingStep object together with the relevant *args and **kwargs
        preprocessing_method = preprocessing.PreprocessingStep(preprocessing_func, **kwargs)
    """

    def __init__(self, method: Callable, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, spectral_data, spectral_axis, *args, **kwargs):
        return self.method(spectral_data, spectral_axis, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(kwargs:{self.kwargs}"

    @final
    def _process_object(self, raman_object: SpectralObject) -> SpectralObject:
        new_raman_object = copy.deepcopy(raman_object)

        preprocessed_spectral_data, preprocessed_spectral_axis = self(
            new_raman_object.spectral_data, new_raman_object.spectral_axis, **self.kwargs)

        new_raman_object.spectral_data = preprocessed_spectral_data
        new_raman_object.spectral_axis = preprocessed_spectral_axis

        return new_raman_object

    @final
    def apply(self, raman_objects: Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]) -> \
            Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]:
        """
        Applies the defined preprocessing method on the Raman spectroscopic objects provided.

        The single point-of-contact method of :class:`ramanspy.preprocessing.PreprocessingStep` instances.

        Method is applied on each data container instance provided individually.


        Parameters
        ----------
        raman_objects : Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The objects to preprocess, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].


        Returns
        -------
        Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The preprocessed objects, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].


        .. note:: When more than one class:`ramanspy.SpectralContainer` is passed, preprocessing methods are applied individually for each instance passed.


        Example
        ----------
    
        .. code::

            # once a preprocessing method is initialised, it can be applied to different Raman data
            preprocessed_data = preprocessing_method.apply(raman_object)
            preprocessed_data = preprocessing_method.apply([raman_object, raman_spectrum, raman_image])
            preprocessed_data = preprocessing_method.apply([raman_object, raman_spectrum], raman_object, [raman_spectrum, raman_image])
        """
        if isinstance(raman_objects, list):
            return [self.apply(raman_object) for raman_object in raman_objects]
        else:
            return self._process_object(raman_objects)
