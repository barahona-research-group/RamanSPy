"""
Generate synthetic endmembers
================================

In this example, we will use `RamanSPy` to generate synthetic endmember signatures.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore

import ramanspy as rp

# Generate synthetic endmembers
endmembers = rp.synth.generate_endmembers(5, 1000, realistic=True)

rp.plot.spectra(endmembers, plot_type='single stacked')
rp.plot.show()

