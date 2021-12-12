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

sys.path.insert(0, os.path.abspath(".."))
print("Path:", sys.path)


# -- Project information -----------------------------------------------------

project = "siibra-python"
copyright = "2020-2021, Forschungszentrum Juelich GmbH"
author = "Big Data Analytics Group, Institute of Neuroscience and Medicine, Forschungszentrum Juelich GmbH"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    #"nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "m2r2",
]
# autosummary_generate = True
autoapi_type = "python"
autoapi_dirs = [os.path.join(os.path.abspath(".."), "siibra")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# example gallery
sphinx_gallery_conf = {
     'examples_dirs': [
         '../examples/snippets/01_atlases_and_parcellations',
         '../examples/snippets/02_maps_and_templates',
         '../examples/snippets/03_data_features',
         '../examples/snippets/04_locations',
         '../examples/snippets/05_anatomical_assignment'
     ],
     'gallery_dirs': [
         'examples/01_atlases_and_parcellations',  
         'examples/02_maps_and_templates',  
         'examples/03_data_features',  
         'examples/04_locations',  
         'examples/05_anatomical_assignment',  
     ],
     'filename_pattern': r'^.*.py', # which files to execute and include their outputs
     'capture_repr': ('_repr_html_', '__repr__'),
     'within_subsection_order': FileNameSortKey,
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# overriding some styles in a custom CSS
html_css_files = ["siibra.css"]

html_logo = "../images/siibra-python.jpeg"

source_suffix = ['.rst']
