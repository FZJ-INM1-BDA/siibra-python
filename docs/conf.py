# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from sphinx_gallery.sorting import FileNameSortKey
import sphinx_rtd_theme
import sphinx_autopackagesummary

from siibra import __version__ as siibra_version

os.environ['SIIBRA_LOG_LEVEL'] = "ERROR"
sys.path.insert(0, os.path.abspath(".."))
print("Path:", sys.path)


# -- Project information -----------------------------------------------------

project = "siibra-python"
copyright = "2020-2023, Forschungszentrum Juelich GmbH"
author = "Big Data Analytics Group, Institute of Neuroscience and Medicine, Forschungszentrum Juelich GmbH"
language = 'en'
version = siibra_version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    "sphinx_autopackagesummary",
    "autoapi.extension",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_rtd_theme",
    "m2r2",
]

# autoapi options
autoapi_member_order = "groupwise"
autoapi_type = "python"
autoapi_dirs = [os.path.join(os.path.abspath(".."), "siibra")]
autoapi_add_toctree_entry = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members'
]

# sphinx_autopackagesummary options
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# example gallery
sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/01_atlases_and_parcellations",
        "../examples/02_maps_and_templates",
        "../examples/03_data_features",
        "../examples/04_locations",
        "../examples/05_anatomical_assignment",
    ],
    "gallery_dirs": [
        "examples/01_atlases_and_parcellations",
        "examples/02_maps_and_templates",
        "examples/03_data_features",
        "examples/04_locations",
        "examples/05_anatomical_assignment",
    ],
    "filename_pattern": r"^.*.py",  # which files to execute and include their outputs
    "capture_repr": ("_repr_html_", "__repr__"),
    "within_subsection_order": FileNameSortKey,
    "remove_config_comments": True,
    "show_signature": False,
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/legacy"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# overriding some styles in a custom CSS
html_css_files = ["siibra.css"]
# other html details
html_show_sourcelink = False
html_logo = "_static/siibra-python.jpeg"
html_permalinks = False

source_suffix = [".rst"]

autoclass_content = 'both'

# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/3/': None}

# -- Sphinxext configuration --------------------------------------------------
