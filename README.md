# RamanSPy

*RamanSPy* is an open-source Python package for integrative
Raman spectroscopy data analysis.

## Installation

*RamanSPy* has been published on PyPI and can be installed
via pip:

``` console
pip install ramanspy
```

## Code example

Below is a simple example of how *RamanSPy* can be used to
load, preprocess and analyse Raman spectroscopic data. Here, we load a
data file from a commercial Raman instrument; apply a preprocessing
pipeline consisting of spectral cropping, cosmic ray removal, denoising,
baseline correction and normalisation; perform spectral unmixing; and
visualise the results.

``` 
import ramanspy as rp

# load data
image_data = rp.load.witec("<PATH>")

# apply a preprocessing pipeline
pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.misc.Cropper(region=(700, 1800)),
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
    rp.preprocessing.baseline.ASPLS(),
    rp.preprocessing.normalise.MinMax()
])
data = pipeline.apply(image_data)

# perform spectral unmixing
nfindr = rp.analysis.unmix.NFINDR(n_endmembers=5)
amaps, endmembers = nfindr.apply(data)

# plot results
rp.plot.spectra(endmembers)
rp.plot.image(amaps)
rp.plot.show()
```

## Documentation

For more information about the functionalities of the package, refer to
the [documentation](https://ramanspy.readthedocs.io).

## Credits

If you find *RamanSPy* useful, please consider leaving a star on [GitHub](https://github.com/barahona-research-group/RamanSPy).