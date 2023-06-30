from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../../src'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ramanspy'
author = 'Dimitar Georgiev'
copyright = f'{date.today().year}, {author}'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    'sphinx_gallery.load_style',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosectionlabel',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_rtd_theme',
    # "sphinx_autodoc_typehints"
    'sphinx_copybutton'
]

templates_path = ['_templates']
exclude_patterns = []


autosummary_generate = True

# autodoc_typehints = "none"


import warnings

warnings.filterwarnings("ignore")


from sphinx_gallery.sorting import FileNameSortKey


sphinx_gallery_conf = {
    'examples_dirs': ['tutorials', 'examples'],
    'gallery_dirs': ['auto_tutorials', 'auto_examples'],
    'matplotlib_animations': True,
    'nested_sections': False,
    'within_subsection_order': FileNameSortKey,
    'show_signature': False,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'RamanSPy'
html_theme = 'furo'
html_static_path = ['_static']
html_theme_path = ['_themes', ]
html_logo = 'images/raman_logo_transparent.png'

html_theme_options = {
    # "logo_only": True,
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
}
html_show_sphinx = False
plot_rcparams = {'savefig.bbox': 'tight'}
plot_apply_rcparams = True
