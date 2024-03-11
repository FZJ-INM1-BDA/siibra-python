# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey
import sphinx_rtd_theme  # this import must be kept to make sphinx_rtd_theme function
import sphinx_autopackagesummary  # this import must be kept to make autopackagesummary function

os.environ['SIIBRA_LOG_LEVEL'] = "ERROR"
sys.path.insert(0, os.path.abspath(".."))
print("Path:", sys.path)

# -- Project information -----------------------------------------------------

project = "siibra-python"
copyright = "2020-2023, Forschungszentrum Juelich GmbH"
author = "Big Data Analytics Group, Institute of Neuroscience and Medicine, Forschungszentrum Juelich GmbH"
language = 'en'

# -- General configuration ---------------------------------------------------

source_suffix = ['.rst', '.md']

# The master toctree document.
root_doc = 'index'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# overriding some styles in a custom CSS
html_css_files = ["siibra.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/legacy"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_gallery.gen_gallery",  # builds an HTML gallery of examples from any set of Python scripts
    "sphinx.ext.autodoc",  # pull in documentation from docstrings in a semi-automatic way
    'sphinx.ext.autosummary',  # generates function/method/attribute summary lists
    'sphinx.ext.autosectionlabel',  # generates the labels for each section
    'sphinx.ext.intersphinx',  # generate links to the documentation of objects in external projects
    'sphinx.ext.napoleon',  # parse both NumPy and Google style docstrings
    "sphinx_autopackagesummary",  # auto generation of API doc for nested Python packages; uses `autosummary`
    "autoapi.extension",  # "autodoc" style doc wo needing to load/run/import the project
    "IPython.sphinxext.ipython_console_highlighting",  # enables ipython syntax highlighting
    "sphinx_rtd_theme",  # readthedocs theme. Requires import or a clone in _static
    "m2r2",  # converts a markdown file including rst markups to a valid rst format
    "sphinxcontrib.jquery",  # work around for jQuery not being loaded automatically dependency removal from sphinx 7
    "sphinx.ext.graphviz",  # to allow drawing diagrams
    "sphinx.ext.inheritance_diagram",  # creates inheritance diagrams
    "sphinx_copybutton",  # adds a copy button for code fields
    "sphinxcontrib.images",  # adds lightbox to images
    "sphinxcontrib.mermaid",   # embed Mermaid graphs including flowcharts, sequence diagrams, gantt diagrams, etc.
]

run_stale_examples = True
rtds_action_github_token = os.environ.get("GITHUB_TOKEN")  # A GitHub personal access token is required
if rtds_action_github_token:
    extensions.append("rtds_action")
    # rtds action settings
    rtds_action_github_repo = "FZJ-INM1-BDA/siibra-python"  # The name of GitHub repository
    rtds_action_path = ""  # The path where the artifact should be extracted # Note: this is relative to the conf.py file!
    rtds_action_artifact_prefix = "sphinx-docs-built-in-github-"  # The "prefix" used in the `upload-artifact` step of the docs github action
    nbsphinx_execute = 'never'
    run_stale_examples = False  # it will be run at github actions (since /docs/example are in gitignore) and locally but not on readthedocs.
else:
    # create package and class diagrams if they were not created with pyreverse and graphviz (see docs.yml)
    cwd = os.getcwd()
    if cwd.endswith('docs'):
        staticpath = os.path.join(cwd, '_static')
        siibrapath = os.path.join(os.path.split(cwd)[0], 'siibra')
    else:
        staticpath = os.path.join(cwd, 'docs/_static')
        siibrapath = os.path.join(cwd, 'siibra')
    if any(
        svg not in os.listdir(staticpath)
        for svg in ["packages_siibra.svg", "classes_siibra.svg"]
    ):
        from subprocess import run
        run(f"pyreverse -k  -o svg -p siibra {siibrapath} --colorized --all-ancestors --output-directory {staticpath}", shell=False)

# napolean settings
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

# Mappings
intersphinx_mapping = {
    "mainconcepts": ('../concepts.html', None),
    "matplotlib": ("https://matplotlib.org/", None),
    "nilearn": ('https://nilearn.github.io/stable/index.html', None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "python": ("https://docs.python.org/3", None),
}

# autoapi options
autoapi_member_order = "groupwise"
autoapi_type = "python"
autoapi_dirs = [os.path.join(os.path.abspath(".."), "siibra")]
autoapi_add_toctree_entry = False
autoapi_options = [
    'members',
    'undoc-members',
    'special-members',
    'show-inheritance-diagram',
    'show-module-summary',
    'imported-members'
]
autoclass_content = 'both'

graphviz_output_format = 'png'
inheritance_graph_attrs = dict(
    rankdir="LR"
)

# sphinx_autopackagesummary options
autosummary_generate = True

# example gallery details
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
    "show_signature": True,
    "run_stale_examples": run_stale_examples
}

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': "bottom",
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

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = True
html_logo = "_static/siibra-python.jpeg"
html_favicon = "_static/siibra_favicon.ico"
html_permalinks = False
