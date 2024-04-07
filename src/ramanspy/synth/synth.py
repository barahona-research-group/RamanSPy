import numpy as np
import scipy

from .. import core

SCENES = ['chessboard', 'gaussian', 'dirichlet']


def _generate_peak(n_bands, *, amplitude_coef=1, width_coef=1):
    """
    Parameters
    ----------
    n_bands : int
        The number of bands to generate.
    amplitude_coef : float, optional
        The amplitude coefficient of the peak.
    width_coef : float, optional
        The width coefficient of the peak.
    """

    peak_position = np.random.randint(10, n_bands - 10)
    peak_height = np.random.uniform(0.1, 1) * amplitude_coef
    peak_width = np.random.uniform(1, 10) * width_coef

    peak = peak_height * np.sqrt(2 * np.pi) * peak_width * scipy.stats.norm.pdf(np.arange(n_bands), peak_position,
                                                                                peak_width)

    return peak


def generate_spectra(num_spectra, n_bands, *, realistic=False, spectral_axis=None, seed=None):
    """
    Generate synthetic spectra.

    Parameters
    ----------
    num_spectra : int
        The number of spectra to generate.
    n_bands : int
        The number of bands to generate.
    realistic : bool, optional
        Whether to generate 'more realistic' spectra by adding smaller noise peaks.
    spectral_axis : array_like, optional
        The spectral axis to use for the spectra. Should match the number of bands.
    seed : int, optional
        The seed to use for the random number generator.

    Returns
    -------
    spectra : list[ramanspy.Spectrum]
        The generated spectra.


    Examples
    --------

    .. code::

        import ramanspy as rp

        # Generate synthetic spectra
        spectra = rp.synth.generate_spectra(5, 100, realistic=True)

        rp.plot.spectra(spectra)
        rp.plot.show()


    References
    ----------
    Georgiev, D., Fernández-Galiana, A., Pedersen, S.V., Papadopoulos, G., Xie, R., Stevens, M.M. and Barahona, M., 2024. Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders. arXiv preprint arXiv:2403.04526.
    """
    if spectral_axis is not None:
        assert len(spectral_axis) == n_bands, 'The spectral axis should match the number of bands.'
    else:
        spectral_axis = np.arange(n_bands)

    np.random.seed(seed)

    spectra = np.zeros((num_spectra, n_bands))

    # generate random peaks
    for i in range(num_spectra):
        # add main peaks
        num_peaks = np.random.randint(5, 10)
        amplitude_coef = 1 + np.random.beta(1, 3) * 5
        for j in range(num_peaks):
            peak = _generate_peak(n_bands, amplitude_coef=amplitude_coef)
            spectra[i, :] += peak

    if realistic:
        for i in range(num_spectra):
            # add noise peaks
            num_peaks = np.random.randint(50, 100)
            for j in range(num_peaks):
                peak = _generate_peak(n_bands, amplitude_coef=1/3, width_coef=2)
                spectra[i, :] += peak

    spectra = [core.Spectrum(spectrum, spectral_axis) for spectrum in spectra]

    return spectra


def mix(
        endmembers,
        abundances,
        *,
        mixture_mode='linear',
        noise=False,
        noise_amplitude=0.1,
        baseline=False,
        baseline_amplitude=2,
        baseline_probability=0.25,
        cosmic_spikes=False,
        cosmic_spike_amplitude=5,
        cosmic_spikes_probability=0.1,
        seed=None):
    """
    Create mixtures based on the given endmembers and abundances.

    Parameters
    ----------
    endmembers : list[Spectrum]
        The underlying endmembers to mix.
    abundances : array_like
        The underlying fractional abundance scene.
    mixture_mode : {'linear', 'nonlinear'}, optional
        The mixing mode to use. Default is 'linear'.
    noise : bool, optional
        Whether to add noise to the image. Default is False.
    noise_amplitude : float, optional
        The amplitude of the noise to add. Default is 0.1.
    baseline : bool, optional
        Whether to add a baseline to the image. Default is False.
    baseline_amplitude : float, optional
        The amplitude of the baseline to add. Default is 2.
    baseline_probability : float, optional
        The probability of adding a baseline to a pixel. Default is 0.25.
    cosmic_spikes : bool, optional
        Whether to add cosmic spikes to the image. Default is False.
    cosmic_spike_amplitude : float, optional
        The amplitude of the cosmic spikes to add. Default is 5.
    cosmic_spikes_probability : float, optional
        The probability of adding a cosmic spike to a pixel. Default is 0.1.
    seed : int, optional
        The seed to use for the random number generator.

    Returns
    -------
    mixtures : array_like
        The mixed spectra.


    References
    ----------
    Georgiev, D., Fernández-Galiana, A., Pedersen, S.V., Papadopoulos, G., Xie, R., Stevens, M.M. and Barahona, M., 2024. Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders. arXiv preprint arXiv:2403.04526.
    """
    spectral_axis = endmembers[0].spectral_axis

    endmembers = np.array([endmember.spectral_data for endmember in endmembers])
    assert abundances.shape[-1] == endmembers.shape[
        0], 'The number of endmembers and the number of abundance maps must match.'

    np.random.seed(seed)

    if mixture_mode == 'linear':
        mixtures = abundances @ endmembers
    elif mixture_mode == 'bilinear':
        endmembers_multiplied = abundances[..., :, None] * endmembers[:, ...]

        r, c = np.triu_indices(abundances.shape[-1], k=1)

        linear_part = np.sum(endmembers_multiplied, axis=-2)
        bilinear_part = np.sum(endmembers_multiplied[..., r, :] * endmembers_multiplied[..., c, :], axis=-2)

        mixtures = linear_part + bilinear_part
    else:
        raise ValueError()

    noise_to_add = np.zeros_like(mixtures)
    if noise:
        noise_to_add = np.random.normal(0, noise_amplitude, mixtures.shape)

    baseline_to_add = np.zeros_like(mixtures)
    if baseline:
        # generate a polynomial baseline
        baseline_function = np.arctan(np.linspace(0, 3.14, mixtures.shape[-1])) * baseline_amplitude
        baseline_indices = np.random.rand(*mixtures.shape[:-1]) < baseline_probability
        baseline_to_add[baseline_indices, ...] = baseline_function * np.random.rand()

    cosmic_spikes_to_add = np.zeros_like(mixtures)
    if cosmic_spikes:
        # generate cosmic spikes at random positions
        flatten_spikes_to_add = cosmic_spikes_to_add.reshape(-1, mixtures.shape[-1])
        for i in range(flatten_spikes_to_add.shape[0]):
            if np.random.rand() < cosmic_spikes_probability:
                spike_position = np.random.randint(2, mixtures.shape[-1] - 2)
                spike_height = np.random.uniform(0.75, 1.25) * cosmic_spike_amplitude
                flatten_spikes_to_add[i, spike_position] = spike_height

        cosmic_spikes_to_add = flatten_spikes_to_add.reshape(mixtures.shape)

    mixtures = mixtures + noise_to_add + baseline_to_add + cosmic_spikes_to_add

    mixtures = core._create_data(mixtures, spectral_axis)

    return mixtures


def generate_abundance_scene(size, num_endmembers, scene_type, *, seed=None):
    """
    Parameters
    ----------
    size : int
        The size of the abundance scene. Assumes a square scene (i.e. size x size).
    num_endmembers : int
        The number of endmembers to use.
    scene_type : {'chessboard', 'gaussian', 'dirichlet'}
        The type of scene to generate.
    seed : int, optional
        The seed to use for the random number generator.

    Returns
    -------
    image : array_like, shape (size, size, num_endmembers)
        The generated abundance image.


    References
    ----------
    Georgiev, D., Fernández-Galiana, A., Pedersen, S.V., Papadopoulos, G., Xie, R., Stevens, M.M. and Barahona, M., 2024. Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders. arXiv preprint arXiv:2403.04526.
    """
    assert scene_type in SCENES, 'The mode must be one of {}'.format(SCENES)

    np.random.seed(seed)

    image = np.zeros((size, size, num_endmembers))

    if scene_type == SCENES[0]:
        split = size // num_endmembers
        for i in range(split):
            for j in range(split):
                e = np.random.choice(num_endmembers)
                image[i * split:(i + 1) * split, j * split:(j + 1) * split, e] = 1

    elif scene_type == SCENES[2]:
        for i in range(size):
            for j in range(size):
                rand_split = np.random.dirichlet(np.ones(num_endmembers), size=1)
                image[i, j, :] = rand_split

    elif scene_type == SCENES[1]:
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        centers = [(size * (i + 1) // (num_endmembers + 1), size * (i + 1) // (num_endmembers + 1)) for i in
                   range(num_endmembers)]
        for i, center in enumerate(centers):
            dist_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
            image[:, :, i] = np.exp(-dist_squared / (2 * (size // num_endmembers) ** 2))

        image = image / np.sum(image, axis=-1, keepdims=True)

    else:
        raise ValueError('Invalid mode')

    return image


def generate_mixture_image(
        num_endmembers,
        num_spectral_bands,
        image_size,
        image_type,
        *,
        mixture_mode='linear',
        realistic_endmembers=False,
        noise=False,
        noise_amplitude=0.1,
        baseline=False,
        baseline_amplitude=2,
        baseline_probability=0.25,
        cosmic_spikes=False,
        cosmic_spike_amplitude=5,
        cosmic_spikes_probability=0.1,
        seed=None):
    """
    Generate a synthetic image dataset.

    Parameters
    ----------
    num_endmembers : int
        The number of endmembers to use.
    num_spectral_bands : int
        The number of spectral bands to use.
    image_size : int
        The size of the image to generate. Assumes a square image.
    image_type : {'chessboard', 'gaussian', 'dirichlet'}
        The type of image to generate.
    mixture_mode : {'linear', 'bilinear'}, optional
        The type of mixture to generate. Default is 'linear'.
    realistic_endmembers : bool, optional
        Whether to use realistic endmembers. Default is False.
    noise : bool, optional
        Whether to add noise to the image. Default is False.
    noise_amplitude : float, optional
        The amplitude of the noise to add. Default is 0.1.
    baseline : bool, optional
        Whether to add a baseline to the image. Default is False.
    baseline_amplitude : float, optional
        The amplitude of the baseline to add. Default is 2.
    baseline_probability : float, optional
        The probability of adding a baseline to a pixel. Default is 0.25.
    cosmic_spikes : bool, optional
        Whether to add cosmic spikes to the image. Default is False.
    cosmic_spike_amplitude : float, optional
        The amplitude of the cosmic spikes to add. Default is 5.
    cosmic_spikes_probability : float, optional
        The probability of adding a cosmic spike to a pixel. Default is 0.1.
    seed : int, optional
        The seed to use for the random number generator.

    Returns
    -------
    mixture : SpectralImage
        The generated mixture.
    endmembers : list[Spectrum]
        The generated endmembers.
    abundance_image : array_like, shape (image_size, image_size, num_endmembers)
        The generated abundance image.


    Examples
    --------

    .. code::

        import ramanspy as rp

        # Generate synthetic data
        mixture, endmebers, abundance_image = rp.synth.generate_image_dataset(5, 1000, 100, 'chessboard', mixture_mode='linear')


    References
    ----------
    Georgiev, D., Fernández-Galiana, A., Pedersen, S.V., Papadopoulos, G., Xie, R., Stevens, M.M. and Barahona, M., 2024. Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders. arXiv preprint arXiv:2403.04526.
    """

    endmebers = generate_spectra(num_endmembers, num_spectral_bands, realistic=realistic_endmembers, seed=seed)
    abundance_image = generate_abundance_scene(image_size, num_endmembers, image_type, seed=seed)

    mixture = mix(endmebers, abundance_image, mixture_mode=mixture_mode, noise=noise, noise_amplitude=noise_amplitude,
                  baseline=baseline, baseline_amplitude=baseline_amplitude, baseline_probability=baseline_probability,
                  cosmic_spikes=cosmic_spikes, cosmic_spike_amplitude=cosmic_spike_amplitude,
                  cosmic_spikes_probability=cosmic_spikes_probability, seed=seed)

    return mixture, endmebers, abundance_image
