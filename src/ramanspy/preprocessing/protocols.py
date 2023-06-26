from . import Pipeline
from . import denoise, baseline, despike, normalise, misc


def default(normalisation_pixelwise: bool = True) -> Pipeline:
    """
    A basic preprocessing protocol.

    Consists of the following steps:

    - cosmic ray removal with Whitaker-Hayes algorithm;
    - denoising with Savitzky-Golay filter with window length 9 and polynomial order 3;
    - baseline correction with Adaptive Smoothness Penalized Least Squares (asPLS);
    - MinMax normalisation (pixelwise).

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.default()
        preprocessed_data = pipeline.apply(data)
    """
    return Pipeline([
        despike.WhitakerHayes(),
        denoise.SavGol(window_length=9, polyorder=3),
        baseline.ASPLS(),
        normalise.MinMax(pixelwise=normalisation_pixelwise),
    ])


def default_fingerprint(normalisation_pixelwise: bool = True) -> Pipeline:
    """
    Same as :meth:`~ramanspy.preprocessing.protocols.default` but starting with spectral cropping.

    Consists of the following steps:

    - spectral cropping to the fingerprint region (700-1800 cm-1);
    - cosmic ray removal with Whitaker-Hayes algorithm;
    - denoising with Savitzky-Golay filter with window length 9 and polynomial order 3;
    - baseline correction with Adaptive Smoothness Penalized Least Squares (asPLS);
    - MinMax normalisation (pixelwise).

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.default_fingerprint()
        preprocessed_data = pipeline.apply(data)
    """
    pipe = default(normalisation_pixelwise)
    pipe.insert(0, misc.Cropper(region=(700, 1800)))
    return pipe


def articular_cartilage() -> Pipeline:
    """
    A basic preprocessing protocol approximating the one adopted in Bergholt MS et al (2016).

    Consists of the following steps:

    - baseline correction with polynomial fitting of order 2 in the range 700-3600 cm-1;
    - spectral cropping to the fingerprint region (700-1800 cm-1);
    - (Unit) Vector normalisation (pixelwise).
    - cosmic ray removal with Whitaker-Hayes algorithm.

    References
    ----------
    Bergholt MS, St-Pierre JP, Offeddu GS, Parmar PA, Albro MB, Puetzer JL, Oyen ML, Stevens MM. Raman spectroscopy reveals new insights into the zonal organization of native and tissue-engineered articular cartilage. ACS central science. 2016 Dec 28;2(12):885-95.


    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.ARTICULAR_CARTILAGE()
        preprocessed_data = pipeline.apply(data)
    """
    return Pipeline([
        baseline.Poly(poly_order=2, regions=[(700, 3600)]),
        misc.Cropper(region=(700, 1800)),
        normalise.Vector(pixelwise=True),
        despike.WhitakerHayes(),
    ])
