|License| |PyPI version| |Python versions| |Documentation Status|

siibra - Software interface for interacting with brain atlases
==============================================================

Copyright 2020-2023, Forschungszentrum Jülich GmbH

*Authors: Big Data Analytics Group, Institute of Neuroscience and
Medicine (INM-1), Forschungszentrum Jülich GmbH*

.. intro-start

``siibra`` is a Python client to a brain atlas framework that integrates brain parcellations and reference spaces at different spatial scales, and connects them with a broad range of multimodal regional data features. 
It aims to facilitate programmatic and reproducible incorporation of brain parcellations and brain region features from different sources into neuroscience workflows.

    **Note:** *``siibra-python`` is still in development. While care is taken that it works reliably, its API is not yet stable and you may still encounter bugs when using it.*

``siibra`` provides structured access to parcellation schemes in different brain reference spaces, including volumetric reference templates at  macroscopic and microscopic resolutions as well as surface representations. 
It supports both discretely labelled and statistical (probabilistic) parcellation maps, which can be used to assign brain regions to spatial locations and image signals, to retrieve region-specific neuroscience datasets from multiple online repositories, and to sample information from high-resolution image data. 
The datasets anchored to brain regions address features of molecular, cellular and architecture as well as connectivity, and are complemented with live queries to external repositories as well as dynamic extraction from "big" image volumes such as the 20 micrometer BigBrain model.

``siibra`` was developed in the frame of the `Human Brain Project <https://humanbrainproject.eu>`__ for accessing the `EBRAINS
human brain atlas <https://ebrains.eu/service/human-brain-atlas>`__. 
It stores most of its contents as sustainable and open datasets in the `EBRAINS Knowledge Graph <https://kg.ebrains.eu>`__, and is designed to support the `OpenMINDS metadata standards <https://github.com/HumanBrainProject/openMINDS_SANDS>`__. 
Its functionalities include common actions known from the interactive viewer ``siibra explorer`` `hosted at EBRAINS <https://atlases.ebrains.eu/viewer>`__. 
In fact, the viewer is a good resource for exploring ``siibra``\ ’s core functionalities interactively: Selecting different parcellations, browsing and searching brain region hierarchies, downloading maps, identifying brain regions, and accessing multimodal features and connectivity information associated with brain regions. 
Feature queries in ``siibra`` are parameterized by data modality and anatomical location, while the latter could be a brain region, brain parcellation, or location in reference space. 
Beyond the explorative focus of ``siibra-explorer``, the Python library supports a range of data analysis functions suitable for typical neuroscience workflows.

``siibra`` hides much of the complexity that would be required to collect and interact with the individual parcellations, templates and data repositories.
By encapsulating many aspects of interacting with different maps and reference templates spaces, it also minimizes common errors like misinterpretation of coordinates from different reference spaces, confusing label indices of brain regions, or using inconsistent versions of parcellation maps. 
It aims to provide a safe way of using maps defined across multiple spatial scales for reproducible analysis.

.. intro-end

.. getting-started-start

Installation
------------

``siibra`` is available on pypi. 
To install the latest released version, simply run ``pip install siibra``. 
In order to work with the latest version from github, use ``pip install git+https://github.com/FZJ-INM1-BDA/siibra-python.git@main``.

There is also an image based on jupyter:scipy-notebook, which already includes ``siibra``.

.. code-block:: sh

  docker run -dit \
        -p 10000:8888 \
        --rm \
        --name siibra \
        docker-registry.ebrains.eu/siibra/siibra-python:latest

Access to EBRAINS
-----------------

While the core features in ``siibra`` can be accessed without any authentication, siibra can perform dynamic queries to regional datasets stored in the `EBRAINS Knowledge Graph <https://kg.ebrains.eu>`__. 
To use this functionality, you need to obtain an EBRAINS authentication token with a valid EBRAINS user account.
`Registering to EBRAINS <https://ebrains.eu/register/>`__ is easy and free of charge, so we strongly recommend to sign up.
To use your EBRAINS access token in siibra:

1. If you do not yet have an EBRAINS account, register `here <https://ebrains.eu/register>`__.
2. When using siibra, fetch an authentication token by using `siibra.fetch_ebrains_token()`. You will be asked to visit an ebrains login website. Login, and accept the requested detail.

Since tokens are temporary, step 2. needs to be repeated regularly.
If you prefer, you can also create your token by visiting `the EBRAINS authorization endpoint <https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize>`__.
Copy the token, and either store it in the environment variable ``$HBP_AUTH_TOKEN`` or pass it explicitly to ``siibra`` using `siibra.set_ebrains_token()`.
The token is a string sequence with more than 1000 characters, usually starting with with “ey”.

Note that as of now, you need to to step 2 approximately every day to perform EBRAINS data queries.
However, ``siibra`` maintains a local cache on disk, so once retrieved, data features become usable and accessible without refreshing the token.

Documentation & Help
--------------------

``siibra-python``\ ’s documentation is hosted on https://siibra-python.readthedocs.io.
The documentation includes a catalogue of documented code examples that walk you through the different concepts and functionalities.
As a new user, it is recommended to go through these examples - they are easy and will quickly provide you with the right code snippets that get you started.
Furthermore, a set of jupyter notebooks demonstrating more extensive example use cases are maintained in the `siibra-tutorials <https://github.com/FZJ-INM1-BDA/siibra-tutorials>`__ repository.
We are working on a full API documentation of the library. You find the current status on readthedocs, but be aware that it is not yet complete and as up-to-date as the code examples.

If you run into issues, please open a ticket on `EBRAINS support <https://ebrains.eu/support/>`__ or file bugs and
feature requests on `github <https://github.com/FZJ-INM1-BDA/siibra-python/issues>`__.
Please keep in mind that ``siibra-python`` is still in development.
While care is taken to make everything work reliably, the API of the library is not yet stable, and the software is not yet fully tested.

.. getting-started-end

.. contribute-start


How to contribute
-----------------

If you want to contribute to ``siibra``, feel free to fork it and open a pull request with your changes.
You are also welcome to contribute to discussions in the issue tracker and of course to report issues you are facing.
If you find the software useful, please reference this repository URL in publications and derived work.
You can also star the project to show us that you are using it.

.. contribute-end

.. acknowledgments-start

Acknowledgements
----------------

This software code is funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

.. acknowledgments-end

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |PyPI version| image:: https://badge.fury.io/py/siibra.svg
   :target: https://pypi.org/project/siibra/
.. |Python versions| image:: https://img.shields.io/pypi/pyversions/siibra.svg
   :target: https://pypi.python.org/pypi/siibra
.. |Documentation Status| image:: https://readthedocs.org/projects/siibra-python/badge/?version=latest
   :target: https://siibra-python.readthedocs.io/en/latest/?badge=latest
