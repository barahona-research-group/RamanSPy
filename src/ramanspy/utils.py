import numpy as np


def is_aligned(raman_objects):
    unique_shift_axes = [
        np.array(unique) for unique in set(tuple(raman_object.spectral_axis) for raman_object in raman_objects)]

    return len(unique_shift_axes) == 1


def wavelength_to_wavenumber(wavelengths, laser_excitation):
    return 1e7/laser_excitation - 1e7/wavelengths


def wavenumber_to_wavelength(raman_shifts, laser_excitation):
    return 1/(1 / laser_excitation - raman_shifts / 1e7)
