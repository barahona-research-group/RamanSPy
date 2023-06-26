from typing import List, Union

from ..core import SpectralObject
from .Step import PreprocessingStep


class Pipeline:
    """
    Defines a preprocessing pipeline consisting of multiple preprocessing procedures.

    Parameters
    ----------
    pipeline : list[:class:`PreprocessingStep`]
        The preprocessing procedures defining the pipeline.

    Example
    ----------

    .. code::

        from ramanspy import preprocessing

        preprocessing_pipeline = preprocessing.Pipeline([
            preprocessing.PreprocessingStep(some_custom_preprocessing_func, *args, **kwargs),
            preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            preprocessing.normalise.Vector()
        ])
    """

    def __init__(self, pipeline: List[PreprocessingStep]):
        self.pipeline = pipeline

    def __getitem__(self, index: int):
        """
        Get a preprocessing procedure from the pipeline at a specified index.

        Parameters
        ----------
        index : int
            The index at which to get the preprocessing procedure.

        """
        return self.pipeline[index]

    def __setitem__(self, index: int, step: PreprocessingStep):
        """
        Set a preprocessing procedure in the pipeline at a specified index.

        Parameters
        ----------
        index : int
            The index at which to set the preprocessing procedure.
        step : :class:`PreprocessingStep`
            The preprocessing procedure to set in the pipeline.

        """
        self.pipeline[index] = step

    def __delitem__(self, index: int):
        """
        Delete a preprocessing procedure from the pipeline at a specified index.

        Parameters
        ----------
        index : int
            The index at which to delete the preprocessing procedure.

        """
        del self.pipeline[index]

    def __len__(self):
        """
        Get the length of the pipeline.
        """
        return len(self.pipeline)

    def __iter__(self):
        """
        Iterate over the preprocessing procedures in the pipeline.
        """
        return iter(self.pipeline)

    def __repr__(self):
        """
        Get a string representation of the pipeline.
        """
        return '\n'.join([f'{i + 1}. {step}' for i, step in enumerate(self.pipeline)])

    def append(self, step: PreprocessingStep):
        """
        Append a preprocessing procedure to the pipeline.

        Parameters
        ----------
        step : :class:`PreprocessingStep`
            The preprocessing procedure to append to the pipeline.

        """
        self.pipeline.append(step)

    def extend(self, steps: List[PreprocessingStep]):
        """
        Extend the pipeline with multiple preprocessing procedures.

        Parameters
        ----------
        steps : list[:class:`PreprocessingStep`]
            The preprocessing procedures to extend the pipeline with.

        """
        self.pipeline.extend(steps)

    def insert(self, index: int, step: PreprocessingStep):
        """
        Insert a preprocessing procedure into the pipeline at a specified index.

        Parameters
        ----------
        index : int
            The index at which to insert the preprocessing procedure.
        step : :class:`PreprocessingStep`
            The preprocessing procedure to insert into the pipeline.

        """
        self.pipeline.insert(index, step)

    def pop(self, index: int):
        """
        Remove a preprocessing procedure from the pipeline at a specified index.

        Parameters
        ----------
        index : int
            The index at which to remove the preprocessing procedure.

        """
        self.pipeline.pop(index)

    def apply(self, raman_objects: Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]) -> \
            Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]:
        """
        Preprocess Raman spectroscopic data using the initialised pipeline.

        The single point-of-contact method of the :class:`Pipeline` class.

        Parameters
        ----------
        raman_objects : Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The objects to preprocess, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].

        Returns
        -------
        Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The preprocessed objects, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].


        .. note:: The :class:`PreprocessingStep` objects comprising the :class:`Pipeline` instance will be applied sequentially in the order provided during initialisation.


        Example
        ----------

        .. code::

            # once a preprocessing pipeline is initialised, it can be applied to different Raman data just as single PreprocessingStep instances
            preprocessed_data = preprocessing_pipeline.apply(raman_object)
            preprocessed_data = preprocessing_method.apply([raman_object, raman_spectrum, raman_image])
            preprocessed_data = preprocessing_method.apply([raman_object, raman_spectrum], raman_object, [raman_spectrum, raman_image])


        """
        # apply each procedure in the constructed pipeline
        for step in self.pipeline:
            raman_objects = step.apply(raman_objects)

        return raman_objects
