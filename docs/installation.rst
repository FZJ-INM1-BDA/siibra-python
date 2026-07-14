.. _getting-started:


Installation and help
#####################

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
  print(siibra.__version__)

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

Update to latest or install a specific version or branch
--------------------------------------------------------

- Update to latest release:

  .. code-block:: bash

    pip install -U siibra

- Install specific version:

  .. code-block:: bash

    pip install siibra==x.y.z

- Install the head of specific branch:

  .. code-block:: bash

    pip install git+https://github.com/FZJ-INM1-BDA/siibra-python.git@branchname

Creating a python environment
=============================

A separate Python environment is recommended when installing scientific Python
packages. For example, using built-in `venv` package:

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

Alternatively, ``uv`` can create virtual environments and select the Python
version used for the environment. This is useful when a workflow should run with
a specific Python version and faster than `venv` package.

.. code-block:: bash

  uv venv --python 3.11 siibra-env
  source siibra-env/bin/activate
  uv pip install siibra

On Windows:

.. code-block:: powershell

  uv venv --python 3.11 siibra-env
  .\siibra-env\Scripts\activate
  uv pip install siibra

If the requested Python version is not available locally, ``uv`` can install or
download a suitable Python version depending on the local ``uv`` configuration.
For details, see the `uv documentation on creating virtual environments
<https://docs.astral.sh/uv/reference/cli/#uv-venv>`__.


Atlas data and cache
====================

`siibra-python` does not download all atlas data during installation. Atlas
metadata are loaded from configurations, and data are fetched lazily when a
workflow requests them.

Fetched files are stored in the local siibra cache. This avoids repeated
downloads and allows later runs of the same workflow to reuse data that are
already available locally.

The first access to a map, template, or data feature may take longer than later
accesses because the requested data have to be downloaded.

Network access
--------------

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

* :ref:`concepts` introduces the central concepts used in siibra.
* :ref:`examples` provides documented code examples.
* :ref:`api` contains the generated API reference.
* The source code and issue tracker are available from the GitHub repository:
  https://github.com/FZJ-INM1-BDA/siibra-python/issues/new/choose
* You can reach siibra team in `Neurostars<https://neurostars.org/tag/siibra/753>`
  with ``siibra`` tag or in `Matrix space<https://matrix.to/#/#siibra:fz-juelich.de>`
