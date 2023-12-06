from . import Pipeline
from . import denoise, baseline, despike, normalise, misc


def georgiev2023_P1(normalisation_pixelwise: bool = True, fingerprint: bool = True) -> Pipeline:
    """
    The first preprocessing protocol used in the paper by Georgiev et al. (2023) [1]_.

    Consists of the following steps:

    - optional: spectral cropping to the fingerprint region (700-1800 cm-1);
    - cosmic ray removal with Whitaker-Hayes algorithm;
    - denoising with a Gaussian filter;
    - baseline correction with Asymmetric Least Squares;
    - Area under the curve normalisation.

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.
    fingerprint: bool, optional
        Whether to crop the spectra to the fingerprint region (700-1800 cm-1) or not. Default is ``True``.

    References
    ----------

    .. [1] Georgiev, D., Pedersen, S.V., Xie, R., Fernández-Galiana, A., Stevens, M.M. and Barahona, M., 2023. RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis. arXiv preprint arXiv:2307.13650.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.georgiev2023_P1()
        preprocessed_data = pipeline.apply(data)
    """
    pipe = Pipeline([
        despike.WhitakerHayes(),
        denoise.Gaussian(),
        baseline.ASLS(),
        normalise.AUC(pixelwise=normalisation_pixelwise),
    ])

    if fingerprint:
        pipe.insert(0, misc.Cropper(region=(700, 1800)))

    return pipe


def georgiev2023_P2(normalisation_pixelwise: bool = True, fingerprint: bool = True) -> Pipeline:
    """
    The second preprocessing protocol used in the paper by Georgiev et al. (2023) [1]_.

    Consists of the following steps:

    - optional: spectral cropping to the fingerprint region (700-1800 cm-1);
    - cosmic ray removal with Whitaker-Hayes algorithm;
    - denoising with Savitzky-Golay filter with window length 9 and polynomial order 3;
    - baseline correction with Adaptive Smoothness Penalized Least Squares (asPLS);
    - MinMax normalisation.

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.
    fingerprint: bool, optional
        Whether to crop the spectra to the fingerprint region (700-1800 cm-1) or not. Default is ``True``.

    References
    ----------

    .. [1] Georgiev, D., Pedersen, S.V., Xie, R., Fernández-Galiana, A., Stevens, M.M. and Barahona, M., 2023. RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis. arXiv preprint arXiv:2307.13650.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.georgiev2023_P2()
        preprocessed_data = pipeline.apply(data)
    """
    pipe = Pipeline([
        despike.WhitakerHayes(),
        denoise.SavGol(window_length=9, polyorder=3),
        baseline.ASPLS(),
        normalise.MinMax(pixelwise=normalisation_pixelwise),
    ])

    if fingerprint:
        pipe.insert(0, misc.Cropper(region=(700, 1800)))

    return pipe


def georgiev2023_P3(normalisation_pixelwise: bool = True, fingerprint: bool = True) -> Pipeline:
    """
    The third preprocessing protocol used in the paper by Georgiev et al. (2023) [1]_.

    Consists of the following steps:

    - optional: spectral cropping to the fingerprint region (700-1800 cm-1);
    - cosmic ray removal with Whitaker-Hayes algorithm;
    - baseline correction with polynomial fitting of order 3;
    - Vector normalisation.

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.
    fingerprint: bool, optional
        Whether to crop the spectra to the fingerprint region (700-1800 cm-1) or not. Default is ``True``.

    References
    ----------

    .. [1] Georgiev, D., Pedersen, S.V., Xie, R., Fernández-Galiana, A., Stevens, M.M. and Barahona, M., 2023. RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis. arXiv preprint arXiv:2307.13650.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.georgiev2023_P3()
        preprocessed_data = pipeline.apply(data)
    """
    pipe = Pipeline([
        despike.WhitakerHayes(),
        baseline.Poly(poly_order=3),
        normalise.Vector(pixelwise=normalisation_pixelwise),
    ])

    if fingerprint:
        pipe.insert(0, misc.Cropper(region=(700, 1800)))

    return pipe


def bergholt2016() -> Pipeline:
    """
    A basic preprocessing protocol approximating the one adopted in Bergholt MS et al. (2016) [1]_.

    Consists of the following steps:

    - baseline correction with polynomial fitting of order 2 in the range 700-3600 cm-1;
    - spectral cropping to the fingerprint region (700-1800 cm-1);
    - (Unit) Vector normalisation (pixelwise).
    - cosmic ray removal with Whitaker-Hayes algorithm.

    References
    ----------

    .. [1] Bergholt MS, St-Pierre JP, Offeddu GS, Parmar PA, Albro MB, Puetzer JL, Oyen ML, Stevens MM. Raman spectroscopy reveals new insights into the zonal organization of native and tissue-engineered articular cartilage. ACS central science. 2016 Dec 28;2(12):885-95.


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
