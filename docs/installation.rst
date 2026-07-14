.. _getting-started:


Installation and help
#####################

`siibra-python` can be installed from the Python Package Index. It is designed
for use in Python scripts, interactive Python sessions, and notebooks.

Installation
============

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

If you prefer to skip creating a virtual environment, you can install the
package with `pip`:

.. code-block:: bash

  pip install siibra

To check the installation, start Python and import the package:

.. code-block:: python

  import siibra
  print(siibra.__version__)


Update to latest or install a specific version or branch
--------------------------------------------------------

- If you encounter a bug or would like to see if there are a new content or
  features are available, please check https://github.com/FZJ-INM1-BDA/siibra-python/tags.
  If there is an update and you would like to get the latest configuration,
  features, and/or bug fixes, you can update to latest release by:

  .. code-block:: bash

    pip install -U siibra

- Install specific version:

  .. code-block:: bash

    pip install siibra==x.y.z

- Install the head of specific branch:

  .. code-block:: bash

    pip install git+https://github.com/FZJ-INM1-BDA/siibra-python.git@branchname


You might need to run ``python -c "import siibra; siibra.cache.clear()"`` if you
encounter problems after the install to clear the cached files since they may
differ between releases and branches.


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
