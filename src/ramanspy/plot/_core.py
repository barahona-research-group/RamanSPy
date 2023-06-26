from typing import Literal
import numpy as np
from matplotlib import pyplot as plt

SPECTRA_PLOT_TYPES = Literal["single", "separate", "stacked", "single stacked"]


def plot_spectral_stack(intensity_stack, shift_axis=None, plot_axis=plt, **plt_kwargs):
    if shift_axis is not None:
        plot_axis.plot(shift_axis, intensity_stack.T, **plt_kwargs)
    else:
        plot_axis.plot(intensity_stack.T, **plt_kwargs)


def plot_species(species, plot_axis=plt, offset=0, **plt_kwargs):
    if not isinstance(species, list):
        species = [species]

    for datum in species:
        intensity_stack, shift_axis = datum if isinstance(datum, tuple) else (datum, None)
        plot_spectral_stack(intensity_stack.reshape(-1, intensity_stack.shape[-1]) + offset, shift_axis,
                            plot_axis=plot_axis, label=plt_kwargs.pop("label", None), **plt_kwargs)


def plot_species_mean(species, plot_axis=plt, offset=0, dist=True, **plt_kwargs):
    if not isinstance(species, list):
        species = [species]

    # Get all data associated with the species
    intensity_stacks = []
    shift_axes = []
    for datum in species:
        intensity_stack, shift_axis = datum if isinstance(datum, tuple) else (datum, None)

        intensity_stacks.append(intensity_stack.reshape(-1, intensity_stack.shape[-1]))
        shift_axes.append(shift_axis)

    # Check if spectral axes are the same
    if all(shift_axis is None for shift_axis in shift_axes):
        shift_axis = None

    #  TODO: np.unique can throw Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or
    #  ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object'
    #  when creating the ndarray.ar = np.asanyarray(ar)
    #  TypeError: The axis argument to unique is not supported for dtype object
    elif all(isinstance(shift_axis, np.ndarray) for shift_axis in shift_axes) and len(
            np.unique(shift_axes, axis=0)) == 1:

        shift_axis = shift_axes[0]
    else:
        ValueError("Cannot plot mean spectral plot of unaligned spectra. Spectral axis must match.")

    # Combine all data into a single intensity stack of the shape (N_0 + N_1 + ... + N_n, B)
    spectral_data = np.vstack(intensity_stacks)

    mu = np.mean(spectral_data, axis=0)

    label = plt_kwargs.pop("label", None)

    if len(spectral_data.shape) == 2 and spectral_data.shape[0] > 1:
        alpha = plt_kwargs.pop("alpha", 1) * 0.2
        if dist:
            # Plot the shaded region corresponding to the 95% confidence interval

            std = np.std(spectral_data, axis=0)

            # 95% confidence interval
            ci = 1.96 * std / np.sqrt(spectral_data.shape[0])

            if shift_axis is not None:
                plot_axis.fill_between(shift_axis, (mu + ci) + offset, (mu - ci) + offset, alpha=alpha, **plt_kwargs)
            else:
                plot_axis.fill_between(np.arange(spectral_data.shape[-1]), (mu + ci) + offset, (mu - ci) + offset,
                                       alpha=alpha, **plt_kwargs)
        else:
            # Plot all spectra shaded behind the mean

            if shift_axis is not None:
                plot_axis.plot(shift_axis, spectral_data.T + offset, alpha=alpha, **plt_kwargs)
            else:
                plot_axis.plot(spectral_data + offset, alpha=alpha, **plt_kwargs)

    # Plot the mean
    if shift_axis is not None:
        plot_axis.plot(shift_axis, mu + offset, label=label, **plt_kwargs)
    else:
        plot_axis.plot(mu + offset, label=label, **plt_kwargs)


def stacked_plots(plotting_function, genus, *,
                  label=None, color=None, title=None, xlabel=None, ylabel=None, **kwargs):
    fig, axs = plt.subplots(nrows=len(genus), ncols=1, sharex=True, sharey=False)

    for species, ax, color_, label_ in zip(genus, axs, color, label):
        plotting_function(species, plot_axis=ax, color=color_, label=label_, **kwargs)

        if label_ is not None:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 0.5), loc="center left",
                      borderaxespad=0)

            plt.subplots_adjust(right=0.75)

    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    return fig


def offset_plot(plotting_function, genus, *, ax=None,
                label=None, color=None, title=None, xlabel=None, ylabel=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    offset = 0
    for species, color_, label_ in zip(genus, color, label):
        if isinstance(species, list):
            intensity_stacks = [spectral_data[0] if isinstance(spectral_data, tuple) else spectral_data for
                                spectral_data in species]
            species_max_value = max(np.max(intensity_stack) for intensity_stack in intensity_stacks)
            species_min_value = min(np.min(intensity_stack) for intensity_stack in intensity_stacks)
        else:
            species_max_value = np.max(species[0] if isinstance(species, tuple) else species)
            species_min_value = np.min(species[0] if isinstance(species, tuple) else species)

        offset -= 1.1 * species_max_value

        plotting_function(species, offset=offset, color=color_, label=label_, **kwargs)

        offset += species_min_value

    if any(label):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.subplots_adjust(right=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_yticks([])

    return ax


def single_plot(plotting_function, genus, *, ax=None,
                label=None, color=None, title=None, xlabel=None, ylabel=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    for species, color_, label_ in zip(genus, color, label):
        plotting_function(species, color=color_, label=label_, plot_axis=ax, **kwargs)

    if any(label):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.subplots_adjust(right=0.75)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def spectra_plot_wrapper(plotting_function, genus, plot_type, **kwargs):
    if not isinstance(kwargs['label'], list):
        kwargs['label'] = [kwargs['label']] * len(genus)

    if kwargs.get('color', None) is None:
        cmap = plt.cm.get_cmap()  # using matplotlib's default colormap
        kwargs['color'] = cmap(np.linspace(0, 1, len(genus)))
    else:
        kwargs['color'] = kwargs['color'] if isinstance(kwargs['color'], list) else [kwargs['color']] * len(genus)

    if plot_type == "single" or len(genus) == 1:
        return single_plot(plotting_function, genus, **kwargs)

    elif plot_type == "single stacked":
        return offset_plot(plotting_function, genus, **kwargs)

    elif plot_type == "stacked":
        return stacked_plots(plotting_function, genus, **kwargs)

    elif plot_type == "separate":
        axs = []
        for species, color, label in zip(genus, kwargs.pop('color'), kwargs.pop('label')):
            fig, ax = plt.subplots()
            ax = single_plot(plotting_function, [species], label=[label], color=[color], ax=ax, **kwargs)
            axs.append(ax)

        if len(axs) == 1:
            return axs[0]
        else:
            return axs
