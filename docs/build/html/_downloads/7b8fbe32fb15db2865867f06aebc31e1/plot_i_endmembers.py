"""
Generate synthetic spectra
================================

In this example, we will use `RamanSPy` to generate synthetic spectra.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore

import ramanspy as rp

# Generate synthetic spectra
spectra = rp.synth.generate_spectra(5, 1000, realistic=True)

rp.plot.spectra(spectra, plot_type='single stacked')
rp.plot.show()

