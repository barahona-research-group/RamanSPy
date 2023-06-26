"""
Loading WITec data
--------------------------------------

Users can use `RamanSPy` to load MATLAB files as exported from `WITec's Suite FIVE <https://raman.oxinst.com/products/software/witec-software-suite>`_ software.

This can be done through the :meth:`ramanspy.load.witec` method by simply providing the name of the file of interest.
"""
import ramanspy

# %%
# The method itself will parse the data to the correct spectral data container, which you can then use as usual.

# %%
# Loading a single spectrum
raman_spectrum = ramanspy.load.witec("path/to/file/witec_spectrum.mat")

# %%
# Loading Raman image data
raman_image = ramanspy.load.witec("path/to/file/witec_image.mat")

# %%
# Loading volumetric Raman data from a list of Raman image files by stacking them as layers along the z-axis
image_layer_files = ["path/to/file/witec_image_1.mat", ..., "path/to/file/witec_image_n.mat"]
raman_image_stack = [ramanspy.load.witec(image_layer_file) for image_layer_file in image_layer_files]
raman_volume = ramanspy.SpectralVolume.from_image_stack(raman_image_stack)
