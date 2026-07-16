|License| |PyPI version| |doi| |Python versions| |Documentation Status|

==============================================================
siibra - Software interface for interacting with brain atlases
==============================================================

Copyright 2018-2026, Forschungszentrum Jülich GmbH

*Authors: Big Data Analytics Group, Institute of Neuroscience and
Medicine (INM-1), Forschungszentrum Jülich GmbH*

`siibra-python` is the Python interface of the siibra tool suite. It provides
programmatic access to reference atlases, brain-region terminologies, reference
coordinate systems, maps of brain parcellations and regions, locations in the
brain, and linked multimodal data features.

The main content distributed with siibra implements the Multilevel Human Brain
Atlas, linking anatomical concepts with maps and measurements across spatial
scales, from macroscopic MRI and surface spaces to microscopic BigBrain
resources. The package supports scripts, notebooks, and reproducible analysis
workflows.

Key capabilities
================

With `siibra-python`, workflows can:

* find brain areas, parcellations, spaces, templates, and maps,
* define coordinates, point sets, bounding boxes, and image-based regions of
  interest,
* assign locations and regions of interest to brain areas,
* query multimodal data features linked to brain areas or spatial locations,
* fetch image, tabular, and numerical data in Python-friendly formats,
* combine atlas information with common scientific Python tools.

`siibra-python` does not ship atlas datasets as part of the package. Instead,
it uses metadata, configurations, and live queries to access distributed data
resources on demand. Data are fetched lazily and can be cached locally.

Installation
============

Install `siibra-python` from the Python Package Index:

.. code-block:: sh

pip install siibra

To install the current development version of the `v1` branch:

.. code-block:: sh

pip install git+https://github.com/FZJ-INM1-BDA/siibra-python.git@v1

Minimal example
===============

.. code-block:: python

import siibra

parcellation = siibra.get_parcellation("julich")
space = siibra.get_space("mni152")
julich_map = siibra.get_map(
parcellation=parcellation,
space=space,
maptype="statistical",
)

region = siibra.get_region("julich", "hoc1 left")
features = siibra.features.get(region)

print(parcellation)
print(space)
print(julich_map)
print(region)
print(features)

Documentation
=============

The documentation includes installation instructions, a concept guide,
documented examples, frequently asked questions, and the generated API
reference: https://siibra-python.readthedocs.io/


The siibra-explorer web viewer is available at: https://atlases.ebrains.eu/viewer/

The siibra HTTP API is available at: https://siibra-api.apps.ebrains.eu/v3_0/docs

Support and contribution
========================

Usage questions and discussions can be posted on Neurostars using the
`siibra` tag.

Bugs, feature requests, and data requests can be reported on the GitHub issue
tracker: https://github.com/FZJ-INM1-BDA/siibra-python/issues

Code and documentation contributions can be proposed through pull requests.
Atlas content and data features are generally added through configuration files
or live queries rather than by modifying the Python source code directly.

Citation
========

Please cite the version of `siibra-python` used in your work. Version-specific
citation information is provided in `CITATION.cff` and through the Zenodo
record associated with each software release.

The general software DOI for `siibra-python` is:

Timo Dickscheid, Xiaoyun Gui, Ahmet Nihat Simsek, Louisa Köhnen,
Vadim Marcenko, Christian Schiffer, Sebastian Bludau, and Katrin Amunts.
`siibra-python`. Zenodo. doi: 10.5281/zenodo.7885728

When referring to siibra as a software tool suite, the Multilevel Human Brain
Atlas, or the conceptual framework connecting atlases, reference spaces,
locations, and multimodal data features, cite the siibra tool suite paper.

A peer-reviewed version of this manuscript has been accepted for publication in
Nature Methods and will be available soon. Until the final publication details
are available, cite the preprint:

Dickscheid, T., Gui, X., Simsek, A. N., Schiffer, C., Mangin, J.-F.,
Leprince, Y., Jirsa, V., Bjaalie, J. G., Leergaard, T. B., Bludau, S.,
and Amunts, K. Siibra: A software tool suite for realizing a Multilevel
Human Brain Atlas from complex data resources. bioRxiv, 2025.
doi: 10.1101/2025.05.20.655042

Cite atlas content, data features, and datasets used in a workflow in addition
to citing the software.

Versioning
==========

`siibra-python` follows semantic versioning. Given a version number
`MAJOR.MINOR.PATCH`, increments imply:

* `MAJOR`: incompatible API changes,
* `MINOR`: functionality is added in a backward-compatible manner,
* `PATCH`: backward-compatible bug fixes or configuration updates, such as
  new maps or features.

Pre-release versions follow the pattern `x.y.z-alpha.t`. By changing `t`,
different `siibra-configurations` versions are targeted.

License
=======

`siibra-python` is released under the Apache License 2.0.

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
:target: https://opensource.org/licenses/Apache-2.0

.. |PyPI version| image:: https://badge.fury.io/py/siibra.svg
:target: https://pypi.org/project/siibra/

.. |Python versions| image:: https://img.shields.io/pypi/pyversions/siibra.svg
:target: https://pypi.python.org/pypi/siibra

.. |Documentation Status| image:: https://readthedocs.org/projects/siibra-python/badge/?version=v1
:target: https://siibra-python.readthedocs.io/en/v1/?badge=v1

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7885728.svg
:target: https://doi.org/10.5281/zenodo.7885728

Acknowledgements
================
This project receives funding from the European Union’s Horizon Europe Programme
(EBRAINS 2.0 Project, grant agreement 101147319), and recieved funding from the
European Union’s Horizon 2020 Research and Innovation Programme (HBP SGA3, grant
agreement 945539), and EBRAIN-Health (101058516).
