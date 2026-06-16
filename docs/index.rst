==============================================================
siibra - Software interface for interacting with brain atlases
==============================================================

`siibra-python` is the Python interface of the siibra tool suite. It provides
programmatic access to reference atlases, brain-region terminologies, reference
coordinate systems, atlas maps, locations in the brain, and linked multimodal
data features.

The main content distributed with siibra implements the Multilevel Human Brain
Atlas. It links anatomical concepts with maps and measurements across multiple
spatial scales, from macroscopic MRI and surface spaces to microscopic BigBrain
resources. The package supports scripts, notebooks, and reproducible analysis
workflows.

.. grid::

  .. grid-item-card:: :fas:`rocket` Quickstart
    :link: getting-started
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Install ``siibra-python`` and run a minimal atlas query.

  .. grid-item-card:: :material-outlined:`stairs;1.5em` Examples
    :link: examples
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Browse documented code examples for common atlas workflows.

  .. grid-item-card:: :material-outlined:`psychology;1.5em` Use cases
    :link: usecases
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Explore neuroscience workflows ``siibra-python`` can be used in.

  .. grid-item-card:: :fas:`book` Tutorial repository :octicon:`link-external`
    :link: https://github.com/FZJ-INM1-BDA/siibra-tutorials
    :link-type: url
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Tutorial notebooks with no-install options to run for workshops and trainings.


With `siibra-python`, you can:

* find brain areas, parcellations, spaces, templates, and maps,
* define coordinates, point sets, bounding boxes, and image-based regions of
  interest,
* assign locations and regions of interest to brain areas,
* query multimodal data features linked to brain areas or spatial locations,
* fetch image, tabular, and numerical data in Python-friendly formats,
* combine atlas information with common scientific Python tools.

`siibra-python` does not ship the atlas datasets themselves. Instead, it uses
metadata, configurations, and live queries to access distributed data resources
on demand. Data are fetched lazily and can be cached locally, so workflows only
download the elements they use.


siibra toolsuite
================

siibra provides complementary interfaces for different use cases:

* `siibra-explorer` is the interactive web viewer for visual exploration of
  atlases, maps, templates, and linked data resources.
* `siibra-python` is the Python library for scripting, notebooks, analysis,
  and reproducible workflows.
* `siibra-api` exposes atlas functionality through HTTP endpoints for
  application development.

The interfaces share the same conceptual basis: reference atlases,
brain-region terminologies, reference coordinate systems, maps, locations, and
data features.

.. grid::
  .. grid-item-card:: siibra-explorer :octicon:`link-external`
    :link: https://atlases.ebrains.eu/viewer/
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
          :columns: 12 4 4 4

          .. image:: https://raw.githubusercontent.com/FZJ-INM1-BDA/siibra-explorer/refs/heads/master/docs/images/siibra-explorer-square.jpeg

      .. grid-item::
          :columns: 12 8 8 8

          .. div:: sd-font-weight-bold

          Explore atlases in your browser

          Use the web viewer to interactively browse reference spaces,
          templates, parcellation maps, brain areas, and linked multimodal
          data. It is a companion for discovering atlas content before using
          it in Python workflows.

EBRAINS research infrastructure
===============================

.. grid::
  .. grid-item-card:: EBRAINS platform :octicon:`link-external`
    :link:  https://www.ebrains.eu/
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 auto auto
    
    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 1

      .. grid-item::
        :columns: 12 4 4 4

      .. image:: https://search.kg.ebrains.eu/static/img/ebrains_logo.svg

    .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

        Many atlas elements and data features accessible through siibra are
        linked to resources hosted or registered in EBRAINS, supporting
        findable and reusable access to curated neuroscience data.

Documentation overview
======================

The documentation is organized around installation, concepts, examples, and API
reference material. The :ref:`getting-started` page covers installation and
first steps with `siibra-python`. The :ref:`glossary` page explains the
central concepts used throughout the documentation. The :ref:`examples`
section provides documented code examples for common atlas workflows. The
:ref:`api` section contains the generated API reference for classes, functions,
and modules.

.. toctree::
  :hidden:
  :maxdepth: 2

  installation
  usecases
  concepts
  examples
  api
  faq
  howtocite

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Development

  create_preconfiguration
  developer
  contribute
  GitHub Repository <https://github.com/FZJ-INM1-BDA/siibra-python>
