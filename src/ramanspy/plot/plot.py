from __future__ import annotations  # default if Python >= 3.10
import copy
import functools
from datetime import datetime
import itertools
from numbers import Number
from typing import List, get_args, Union, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from . import _core

if TYPE_CHECKING:
    from ..core import SpectralObject, Spectrum


def show():
    """
    Show all the plots.
    """
    plt.show()


def peaks(spectrum: Spectrum,
          *,
          title: str = "Raman spectra",
          xlabel: str = 'Raman shift (cm$^{{{-1}}}$)',
          ylabel: str = 'Intensity (a.u.)',
          color=None,
          height=None,
          threshold=None,
          distance=None,
          prominence=None,
          width=None,
          wlen=None,
          rel_height=0.5,
          plateau_size=None,
          return_peaks: bool = False,
          **plt_kwargs):
    """
    Visualising peaks.

    Parameters
    -----------
    spectrum : Spectrum
        The spectral data to plot.
    title : str, optional
        The title of the plot. Default is ``'Raman spectra'``.
    xlabel : str, optional
        The x-axis label of the plot. Default is ``'Raman shift (cm$^{{{-1}}}$)'``.
    ylabel : str, optional
        The y-axis label of the plot. Default is ``'Intensity (a.u.)'``.
    color : matplotlib color, optional
        The color(s) to use for each plot. Default is ``None``, i.e. the default matplotlib's colormap will be used,
        which is the ``veridis`` colormap.
    height, threshold, distance, prominence, width, wlen, rel_heigh, plateau_size : optional
        Parameter for the `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ method.
        Check the original documentation for more information.
    return_peaks: bool, optional
        Whether to return the peaks and their properties. Default is ``False``.
    **plt_kwargs :
        Additional parameters. Will be passed to the `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_ method.

    Returns
    -------
    matplotlib.axes.Axes :
        The Axes object of the plot.
    numpy.ndarray, optional
        The peaks found in the spectrum. Only returned if ``return_peaks`` is ``True``.
    dict, optional
        The properties of the peaks found in the spectrum. Only returned if ``return_peaks`` is ``True``.


    Examples
    ---------

    .. code::

        import ramanspy as rp

        # plots peaks within a single spectrum
        rp.plot.peaks(spectrum, **kwargs)
    """

    peaks, properties = spectrum.peaks(height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)

    ax = spectra(spectrum, title=title, xlabel=xlabel, ylabel=ylabel, color=color, **plt_kwargs)
    ax.plot(spectrum.spectral_axis[peaks], spectrum.spectral_data[peaks], "xr")

    min_intensity = np.min(spectrum.spectral_data)
    max_intensity = np.max(spectrum.spectral_data)

    for peak in peaks:
        ax.text(spectrum.spectral_axis[peak] - 0.02*spectrum.spectral_axis[-1], spectrum.spectral_data[peak]+0.025*np.abs(max_intensity), int(np.round(spectrum.spectral_axis[peak])))

    ax.set_ylim(min_intensity-0.1*np.abs(min_intensity), max_intensity+0.1*np.abs(max_intensity))

    if return_peaks:
        peaks = spectrum.spectral_axis[peaks]
        return ax, peaks, properties
    else:
        return ax


def spectra(
        spectra: Union[
            NDArray, SpectralObject, List[Union[NDArray, SpectralObject]], List[List[Union[NDArray, SpectralObject]]]],
        wavenumber_axis: NDArray = None,
        *,
        plot_type: _core.SPECTRA_PLOT_TYPES = 'separate',
        title: str = "Raman spectra",
        xlabel: str = 'Raman shift (cm$^{{{-1}}}$)',
        ylabel: str = 'Intensity (a.u.)',
        label: List[str] = None,
        color=None,
        **kwargs
):
    """
    Visualising spectra.

    Parameters
    -----------
    spectra : Union[NDArray, SpectralObject, List[Union[NDArray, SpectralObject]], List[List[Union[NDArray, SpectralObject]]]]
        The spectral data to plot. Can plot a single spectrum and collection(s) of spectra, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].
    wavenumber_axis : numpy.ndarray, optional
        The shift axis of the data provided. Only used if ``spectra`` contains data which is not a spectral container.
        Must match for all spectra provided.
    plot_type : {"single", "separate", "stacked", "single stacked"}, optional
        The type of the plot. Default is ``'separate'``.

        - ``'single'`` - groups are plotted in the same plot;
        - ``'separate'` - groups are plotted in individual plots;
        - ``'stacked'`` - groups are plotted in individual plots, stacked on top of each other;
        - ``'single stacked'`` - groups are plotted in the same plot, stacked on top of each other.

    title : str, optional
        The title of the plot. Default is ``'Raman spectra'``.
    xlabel : str, optional
        The x-axis label of the plot. Default is ``'Raman shift (cm$^{{{-1}}}$)'``.
    ylabel : str, optional
        The y-axis label of the plot. Default is ``'Intensity (a.u.)'``.
    label : Union[str, List[str]], optional
        The label(s) of the spectral group(s) provided. Must match ``spectra``. Default is ``None``, i.e. no labels.
    color : Union[str, List[str]], optional
        The color(s) to use for each plot. Default is ``None``, i.e. the default matplotlib's colormap will be used,
        which is the ``veridis`` colormap.
    **kwargs :
        Additional parameters. Will be passed to the `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_ method.


    Returns
    -------
    matplotlib.axes.Axes or List[matplotlib.axes.Axes] :
        The Axes object(s) of the plot(s).


    Examples
    ---------

    .. code::

        import ramanspy as rp

        # plots a single spectrum
        rp.plot.spectra(spectrum)

        # plots all spectra within a SpectralContainer instance as a single group
        rp.plot.spectra(spectral_object)

        # plots 3 spectra as individual groups
        rp.plot.spectra([spectrum_1, spectrum_2, spectrum_3])

        # plots 2 groups of spectra (size does not need to be the same across groups)
        rp.plot.spectra([spectrum_1, spectrum_2, spectrum_3], [spectrum_4, spectrum_5])

        # plots 3 spectra as a single group
        rp.plot.spectra([[spectrum_1, spectrum_2, spectrum_3]])
    """
    options = get_args(_core.SPECTRA_PLOT_TYPES)
    if plot_type not in options:
        raise ValueError(f"Plot type '{plot_type}' is not supported. Available plot types are {options}")

    if not isinstance(spectra, list):
        spectra = [spectra]

    if isinstance(label, list):
        assert len(label) == len(spectra)

    genus_data = []
    for species in spectra:
        if isinstance(species, list):
            species_data = [
                (species_obj.spectral_data, species_obj.spectral_axis) if hasattr(species_obj, "spectral_data") else (
                species_obj, wavenumber_axis) for species_obj in species]
            genus_data.append(species_data)
        else:
            genus_data.append((species.spectral_data, species.spectral_axis) if hasattr(species, "spectral_data") else (
            species, wavenumber_axis))

    return _core.spectra_plot_wrapper(_core.plot_species, genus_data, label=label, plot_type=plot_type,
                                      title=title, xlabel=xlabel, ylabel=ylabel, color=color, **kwargs)


def mean_spectra(
        spectra: Union[
            NDArray, SpectralObject, List[Union[NDArray, SpectralObject]], List[List[Union[NDArray, SpectralObject]]]],
        wavenumber_axis: NDArray = None,
        *,
        plot_type: _core.SPECTRA_PLOT_TYPES = 'separate',
        dist: bool = True,
        title: str = "Raman spectra",
        xlabel: str = 'Raman shift (cm$^{{{-1}}}$)',
        ylabel: str = 'Intensity (a.u.)',
        label: List[str] = None,
        color=None,
        **kwargs
):
    """
    Visualising spectral distributions.

    Parameters
    -----------
    spectra : Union[NDArray, SpectralObject, List[Union[NDArray, SpectralObject]], List[List[Union[NDArray, SpectralObject]]]]
        The spectral data to plot. Can plot a single spectrum and collection(s) of spectra, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].
    wavenumber_axis : numpy.ndarray, optional
        The shift axis of the data provided. Only used if ``spectra`` contains data which is not a spectral container.
        Must match for all spectra provided.
    plot_type : {"single", "separate", "stacked", "single stacked"}, optional
        The type of the plot. Default is ``'separate'``.

        - ``'single'`` - groups are plotted in the same plot;
        - ``'separate'` - groups are plotted in individual plots;
        - ``'stacked'`` - groups are plotted in individual plots, stacked on top of each other;
        - ``'single stacked'`` - groups are plotted in the same plot, stacked on top of each other.

    dist : bool, optional
        If ``dist=True``, the method will plot the mean spectrum and a 95% confidence interval of each distribution (default).
        If ``dist=False``, it will plot the mean and the individual spectra comprising each distribution.
    title : str, optional
        The title of the plot. Default is ``'Raman spectra'``.
    xlabel : str, optional
        The x-axis label of the plot. Default is ``'Raman shift (cm$^{{{-1}}}$)'``.
    ylabel : str, optional
        The y-axis label of the plot. Default is ``'Intensity (a.u.)'``.
    label : Union[str, List[str]], optional
        The label(s) of the spectral group(s) provided. Default is ``None``, i.e. no labels.
    color : Union[str, List[str]], optional
        The color(s) to use for each plot. Default is ``None``, i.e. the default matplotlib's colormap will be used,
        which is the ``veridis`` colormap.
    **kwargs :
        Additional parameters. Will be passed to the `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_ method.

    Returns
    -------
    matplotlib.axes.Axes or List[matplotlib.axes.Axes] :
        The Axes object(s) of the plot(s).


    Examples
    ---------

    .. code::

        import ramanspy as rp

        # plots the distributions of 2 groups of spectra
        rp.plot.mean_spectra([spectrum_1, spectrum_2, spectrum_3], [spectrum_4, spectrum_5])

    """
    options = get_args(_core.SPECTRA_PLOT_TYPES)
    assert plot_type in options, f"Plot type '{plot_type}' is not supported. Available plot types are {options}"

    if not isinstance(spectra, list):
        spectra = [spectra]

    if isinstance(label, list):
        assert len(label) == len(spectra)

    if isinstance(spectra, list) and all(not isinstance(spectrum, list) for spectrum in spectra):
        spectra = [spectra]

    genus_data = []
    for species in spectra:
        if isinstance(species, list):
            species_data = [
                (species_obj.spectral_data, species_obj.spectral_axis) if hasattr(species_obj, "spectral_data") else (
                species_obj, wavenumber_axis) for species_obj in species]
            genus_data.append(species_data)
        else:
            genus_data.append((species.spectral_data, species.spectral_axis) if hasattr(species, "spectral_data") else (
            species, wavenumber_axis))

    return _core.spectra_plot_wrapper(_core.plot_species_mean, genus_data, label=label, plot_type=plot_type,
                                      title=title, xlabel=xlabel, ylabel=ylabel, dist=dist, color=color, **kwargs)


def scalable(plotting_function):
    @functools.wraps(plotting_function)
    def wrap(data_to_plot, **kwargs):
        if not isinstance(data_to_plot, list):
            data_to_plot = [data_to_plot]

        if kwargs.get("color", None) is None:
            cmap = plt.cm.get_cmap()  # using matplotlib's default colormap
            kwargs["color"] = list(cmap(np.linspace(0, 1, len(data_to_plot))))

        save_to = kwargs.get("save_to", None)
        if save_to is not None:
            kwargs["save_to"] = f"{save_to}_{datetime.now().strftime('%Y-%m-%d %H%M%S')}"

        kwargs = {k: itertools.repeat(v, len(data_to_plot)) if not isinstance(v, list) else v for k, v in
                  kwargs.items()}

        outputs = []
        for input_params in zip(data_to_plot, *kwargs.values()):
            new_kwargs = {k: v for k, v in zip(kwargs.keys(), input_params[1:])}
            output = plotting_function(input_params[0], **new_kwargs)

            outputs.append(output)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    return wrap


@scalable
def image(
        image: Union[NDArray, List[NDArray]],
        *,
        ax=None,
        threshold: Union[Number, List[Number]] = None,
        title: Union[str, List[str]] = "Raman image",
        xlabel: Union[str, List[str]] = None,
        ylabel: Union[str, List[str]] = None,
        cbar: Union[bool, List[bool]] = True,
        cbar_label: Union[str, List[str]] = "Peak intensity",
        color=None,
        **plt_kwargs
):
    """
    Visualising imaging Raman data.

    If more than one image slice is provided, they will be plotted in separate plots using the corresponding parameters given.

    Parameters
    -----------
    image : Union[NDArray, List[NDArray]]
        2D array(s) corresponding to the spectral slice(s) to visualise.
    threshold : Union[Number, List[Number]], optional
        If provided, all values less than the given threshold will be discarded, i.e. set to the minimum value in the data.
    title : Union[str, List[str]], optional
        The plot title(s) to use for each plot. Default is ``'Raman image'``.
    xlabel : Union[str, List[str]], optional
        The x-axis label(s) to use for each plot. Default is ``None``, i.e. no label.
    ylabel : Union[str, List[str]], optional
        The y-axis label(s) to use for each plot. Default is ``None``, i.e. no label.
    cbar : Union[bool, List[bool]], optional
        Whether to include a colorbar or not in each plot. Default is ``True``.
    cbar_label : Union[str, List[str]], optional
        If ``cbar=True``, the colorbar label(s) to use for each plot. Default is ``'Peak intensity'``.
    color : Union[Matplotlib color, List[Matplotlib color]], optional
        The color(s) to use for each plot. Default is ``None``, i.e. the default matplotlib's colormap will be used,
        which is the ``veridis`` colormap.
    **plt_kwargs : keyword arguments, optional
        Additional parameters. Will be passed to the `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_ method.
        Each parameter can be given by single instance or as a list of instances for each plot.

    Returns
    -------
    matplotlib.axes.Axes or List[matplotlib.axes.Axes] :
        The Axes object(s) of the plot(s).


    Examples
    ---------

    .. code::

        import ramanspy as rp

        # plot single image slice
        ax = rp.plot.image(raman_image.band(1500))

        # visualising
        plt.show()

        # saving
        ax.figure.savefig('...')


        # plot a list of image slices
        ax = rp.plot.image([raman_image.band(1500), raman_image.band(2500)])

        # plot a list of image slices with shared parameters
        ax = rp.plot.image([raman_image.band(1500), raman_image.band(2500)], title="Spectral slice", cbar=True, ...)

        # plot a list of image slices with different parameters
        ax = rp.plot.image([raman_image.band(1500), raman_image.band(2500)], title=["Spectral slice A", "Spectral slice B"], cbar=[True, False], ...)
    """
    if threshold is not None:
        image = copy.deepcopy(image)
        image[image < threshold] = np.min(image)

    if ax is None:
        fig, ax = plt.subplots()

    white = [1, 1, 1, 0]
    cmap = LinearSegmentedColormap.from_list('', [white, color])

    im = ax.imshow(image, cmap=cmap, **plt_kwargs)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, cax=cax, ticks=[image.min(), image.max()], cmap=cmap, label=cbar_label)

    # draw box around volume
    edges_kw = dict(color='black', linewidth=.5)

    # draw box around volume
    edges_kw = dict(color='black', linewidth=.5)

    ax.plot([-1, -1], [-1, image.shape[0]], **edges_kw)
    ax.plot([image.shape[1], image.shape[1]], [-1, image.shape[0]], **edges_kw)
    ax.plot([-1, image.shape[1]], [-1, -1], **edges_kw)
    ax.plot([-1, image.shape[1]], [image.shape[0], image.shape[0]], **edges_kw)

    # set ax parameters
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.spines.values(), color=None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


@scalable
def volume(
        volume: Union[NDArray, List[NDArray]],
        *,
        ax=None,
        threshold: Number = None,
        color=None,
        title: Union[str, List[str]] = "Raman volume",
        xlabel: Union[str, List[str]] = None,
        ylabel: Union[str, List[str]] = None,
        zlabel: Union[str, List[str]] = None,
        cbar: Union[bool, List[bool]] = True,
        cbar_label: Union[str, List[str]] = "Peak intensity",
        **plt_kwargs
):
    """
    Visualising volumetric Raman data.

    If more than one volume slice is provided, they will be plotted in separate plots using the corresponding parameters given.

    Parameters
    -----------
    volume : Union[NDArray, List[NDArray]]
        3D array(s) corresponding to the spectral slice(s) to visualise.
    threshold : Union[Number, List[Number]], optional
        If provided, all values less than the given threshold will be discarded, i.e. set to the minimum value in the data.
    title : Union[str, List[str]], optional
        The plot title(s) to use for each plot. Default is ``'Raman volume'``.
    xlabel : Union[str, List[str]], optional
        The x-axis label(s) to use for each plot. Default is ``None``, i.e. no label.
    ylabel : Union[str, List[str]], optional
        The y-axis label(s) to use for each plot. Default is ``None``, i.e. no label.
    zlabel : Union[str, List[str]], optional
        The z-axis label(s) to use for each plot. Default is ``None``, i.e. no label.
    cbar : Union[bool, List[bool]], optional
        Whether to include a colorbar or not in each plot. Default is ``True``.
    cbar_label : Union[str, List[str]], optional
        If ``cbar=True``, the colorbar label(s) to use for each plot. Default is ``'Peak intensity'``.
    color : Union[Matplotlib color, List[Matplotlib color]], optional
        The color(s) to use for each plot. Default is ``None``, i.e. the default matplotlib's colormap will be used,
        which is the ``veridis`` colormap.
    **plt_kwargs : keyword arguments, optional
        Additional parameters. Will be passed to the `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_ method.
        Each parameter can be given by single instance or as a list of instances for each plot.


    Returns
    -------
    matplotlib.axes.Axes or List[matplotlib.axes.Axes] :
        The Axes object(s) of the plot(s).

    Examples
    ---------

    .. code::

        import ramanspy as rp

        # plot single volume slice
        ax = rp.plot.volume(raman_volume.band(1500))

        # visualising
        plt.show()

        # saving
        ax.figure.savefig('...')


        # plot a list of volume slices
        rp.plot.volume([raman_volume.band(1500), raman_volume.band(2500)])

        # plot a list of volume slices with shared parameters
        rp.plot.volume([raman_volume.band(1500), raman_volume.band(2500)], title="Spectral slice", cbar=True, ...)

        # plot a list of volume slices with different parameters
        rp.plot.volume([raman_volume.band(1500), raman_volume.band(2500)], title=["Spectral slice A", "Spectral slice B"], cbar=[True, False], ...)

    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # get colormap
    white = [1, 1, 1, 0]
    cmap = LinearSegmentedColormap.from_list('', [white, color])

    X, Y, Z = np.mgrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]

    # remove pixels below threshold value (e.g. remove background)
    if threshold is not None:
        volume = copy.deepcopy(volume)
        volume[volume < threshold] = np.min(volume)

    # plot
    plot = ax.scatter(X, Y, Z, c=volume, cmap=cmap, **plt_kwargs)

    # add colorbar
    if cbar:
        ax.figure.colorbar(plot, ax=ax, shrink=0.5, ticks=[volume.min(), volume.max()], cmap=cmap, label=cbar_label)

    # draw box around volume
    edges_kw = dict(color='black', linewidth=.5)

    ax.plot([-1, -1], [-1, volume.shape[1]], -1, **edges_kw)
    ax.plot([volume.shape[0], volume.shape[0]], [-1, volume.shape[1]], -1, **edges_kw)
    ax.plot([-1, volume.shape[0]], [-1, -1], -1, **edges_kw)
    ax.plot([-1, volume.shape[0]], [volume.shape[1], volume.shape[1]], -1, **edges_kw)

    ax.plot([-1, -1], [-1, volume.shape[1]], volume.shape[2], **edges_kw)
    ax.plot([volume.shape[0], volume.shape[0]], [-1, volume.shape[1]], volume.shape[2], **edges_kw)
    ax.plot([-1, volume.shape[0]], [-1, -1], volume.shape[2], **edges_kw)
    ax.plot([-1, volume.shape[0]], [volume.shape[1], volume.shape[1]], volume.shape[2], **edges_kw)

    ax.plot([-1, -1], [-1, -1], [-1, volume.shape[2]], **edges_kw)
    ax.plot([-1, -1], [volume.shape[1], volume.shape[1]], [-1, volume.shape[2]], **edges_kw)
    ax.plot([volume.shape[0], volume.shape[0]], [-1, -1], [-1, volume.shape[2]], **edges_kw)
    ax.plot([volume.shape[0], volume.shape[0]], [volume.shape[1], volume.shape[1]], [-1, volume.shape[2]], **edges_kw)

    # set ax parameters
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.grid(False)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.xaxis.line.set_color(white)
    ax.yaxis.line.set_color(white)
    ax.zaxis.line.set_color(white)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(60, 0)
    ax.set_box_aspect(volume.shape)

    return ax


def peak_dist(
        spectra: Union[SpectralObject, List[SpectralObject], List[List[SpectralObject]]],
        band: Number,
        *,
        ax=None,
        labels: List[str] = None,
        title: str = "Peak distribution",
        ylabel: str = 'Intensity (a.u.)',
        **kwargs
):
    """
    Visualising peak distributions as barplots.

    Error bars represent one standard deviation of uncertainty.

    Parameters
    -----------
    spectra : Union[SpectralObject, List[SpectralObject], List[List[SpectralObject]]]
        The spectral data to plot, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].
    band : Number
        The spectral band of interest.
    ax : Matplotlib Axes object, optional
        If provided, the plot will be added to the given Axes instance. Default is ``None``, i.e. a new Axes instance will be created.
    labels : Union[str, List[str]], optional
        The label(s) of the spectral group(s) provided. Must match ``spectra``. Default is ``None``, i.e. no labels.
    title : str, optional
        The title of the plot. Default is ``'Raman spectra'``.
    ylabel : str, optional
        The y-axis label of the plot. Default is ``'Intensity (a.u.)'``.
    **kwargs :
        Additional parameters. Will be passed to the `matplotlib.pyplot.bar <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html>`_ method.

    Returns
    -------
    matplotlib.axes.Axes :
        The Axes object of the plot.

    Examples
    ---------

    .. code::

        import matplotlib.pyplot as plt
        import ramanspy as rp

        # plots the peak distributions at 1500cm^-1 of 2 groups of spectra
        ax = rp.plot.peak_dist([spectrum_1, spectrum_2, spectrum_3], [spectrum_4, spectrum_5], band=1500, labels=["Group A", "Group B"])

        # visualising
        plt.show()  # or rp.plot.show()

        # saving
        ax.figure.savefig('...')

    """
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(spectra[0], list):
        peak_values = [
            np.concatenate([spectral_object.band(band).flatten() for spectral_object in subspectra])
            for subspectra in spectra]
    else:
        peak_values = [
            np.concatenate([spectral_object.band(band).flatten() for spectral_object in spectra])]

    peak_values_means = [np.mean(peak_value) for peak_value in peak_values]
    peak_values_std = [np.std(peak_value) for peak_value in peak_values]

    if labels is None:
        labels = [None] * len(spectra)

    labels = [label if label is not None else 'Unlabelled' for label in labels]
    labels = [f"{labels[i]}\n(n={len(peak_values[i])})" for i in range(len(labels))]

    ax.bar(labels, peak_values_means, yerr=peak_values_std, capsize=10, **kwargs)

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    return ax
