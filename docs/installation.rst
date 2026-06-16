.. _getting-started:

=====================
Installation and help
=====================

`siibra-python` can be installed from the Python Package Index. It is designed
for use in Python scripts, interactive Python sessions, and notebooks.

Installation
============

Install the package with `pip`:

.. code-block:: bash

  pip install siibra

To check the installation, start Python and import the package:

.. code-block:: python

  import siibra
  print(siibra.**version**)

A minimal query can be used to verify that siibra can access its default
configuration:

.. code-block:: python

  import siibra

  parcellation = siibra.get_parcellation("julich")
  space = siibra.get_space("mni152")
  julich_map = siibra.get_map(
  parcellation=parcellation,
  space=space,
  maptype="statistical",
  )

  print(parcellation)
  print(space)
  print(julich_map)

Python environment
==================

A separate Python environment is recommended when installing scientific Python
packages. For example, using `venv`:

.. code-block:: bash

  python -m venv siibra-env
  source siibra-env/bin/activate
  pip install --upgrade pip
  pip install siibra

On Windows, the activation command is different:

.. code-block:: powershell

  python -m venv siibra-env
  .\siibra-env\Scripts\activate
  pip install --upgrade pip
  pip install siibra

Optional dependencies
=====================

Some workflows use additional scientific Python packages, depending on the data
types being processed. Common examples include:

* `nibabel` for neuroimaging image objects,
* `nilearn` for neuroimaging analysis and visualization,
* `pandas` for tabular data,
* `matplotlib` for plotting.

Install optional packages as needed for a workflow. For example:

.. code-block:: bash

  pip install nilearn matplotlib

Data access and cache
=====================

`siibra-python` does not download all atlas data during installation. Atlas
metadata are loaded from configurations, and data are fetched lazily when a
workflow requests them.

Fetched files are stored in the local siibra cache. This avoids repeated
downloads and allows later runs of the same workflow to reuse data that are
already available locally.

The first access to a map, template, or data feature may take longer than later
accesses because the requested data have to be downloaded.

Network access
==============

Many atlas elements and data features point to external resources. A network
connection is therefore required for typical first-time use.

In restricted environments, such as institutional networks or offline analysis
setups, workflows may require cache preparation or a local configuration. See
the development and configuration documentation for details:

* :ref:`developer`
* :doc:`create_preconfiguration`

Getting help
============

The following resources provide further information:

* :ref:`glossary` introduces the central concepts used in siibra.
* :ref:`examples` provides documented code examples.
* :ref:`api` contains the generated API reference.
* The source code and issue tracker are available from the GitHub repository:
  https://github.com/FZJ-INM1-BDA/siibra-python

When reporting a problem, include the installed siibra version, the Python
version, the operating system, and a minimal code example that reproduces the
issue.
