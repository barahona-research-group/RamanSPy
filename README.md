<p align="center">
  <a href="https://ramanspy.readthedocs.io/">
    <img src="https://github.com/barahona-research-group/RamanSPy/blob/1121738ca4b8b64d938b81eefe32059ac33ace8e/docs/source/images/raman_logo_transparent.png" alt="RamanSPy logo"  width="300">
  </a>
</p>

## *RamanSPy*: An open-source package for <ins>Raman</ins> <ins>S</ins>pectroscopy analytics in <ins>Py</ins>thon.

[![Downloads](https://static.pepy.tech/badge/ramanspy)](https://pepy.tech/project/ramanspy)   [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)]([https://github.com/dwyl/esta/issues](https://github.com/barahona-research-group/RamanSPy/issues))


## Key features

![Overview of RamanSPy](docs/source/images/ramanspy_graphical_abstract.png)

- Common data format
- Data loaders
- Preprocessing methods
- Preprocessing pipelining
- Preprocessing protocols
- Analysis methods
- AI & ML integration
- Visualisation tools
- Datasets
- Synthetic data generator
- Metrics
  

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
the [online documentation](https://ramanspy.readthedocs.io).

## Credits

If you use *RamanSPy* for your research, please cite our paper:

[Georgiev, D.; Pedersen, S. V.; Xie, R.; Fernández-Galiana, Á.; Stevens, M. M.; Barahona, M. *RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis*. ACS Analytical Chemistry **2024**, 96(21), 8492-8500, DOI: 10.1021/acs.analchem.4c00383](https://pubs.acs.org/doi/10.1021/acs.analchem.4c00383)

```bibtex
@article{georgiev2024ramanspy,
    title={RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis},
    author={Georgiev, Dimitar and Pedersen, Simon Vilms and Xie, Ruoxiao and Fern{\'a}ndez-Galiana, Alvaro and Stevens, Molly M and Barahona, Mauricio},
    journal={Analytical Chemistry},
    volume={96},
    number={21},
    pages={8492-8500},
    year={2024},
    doi={10.1021/acs.analchem.4c00383}
}
```

Also, if you find *RamanSPy* useful, please consider leaving a star on [GitHub](https://github.com/barahona-research-group/RamanSPy).
