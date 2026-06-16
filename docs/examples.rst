.. _examples:

========
Examples
========

This section provides documented code examples for common `siibra-python`
workflows. The examples are organized by the main concepts used throughout the
package: reference atlases, brain-region terminologies, maps, reference spaces,
locations, anatomical assignment, and multimodal data features.

The examples are intended to complement the concept guide and the generated API
reference. For conceptual background, see :ref:`glossary`. For function and
class details, see :ref:`api`.

.. grid::
  
  .. grid-item-card:: :material-outlined:`account_tree;2em` Atlases and brain-region terminologies
    :link: examples/01_atlases_and_parcellations/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Find reference atlases, parcellations, and brain areas. These examples
    introduce the anatomical concepts used to organize atlas content.

  .. grid-item-card:: :material-outlined:`map;2em` Maps, spaces, and templates
    :link: examples/02_maps_and_templates/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Access annotation maps and reference templates in supported coordinate
    systems. These examples cover labelled and statistical maps, spaces, and
    image data retrieval.


  .. grid-item-card:: :material-outlined:`query_stats;2em` Multimodal data features
    :link: examples/03_data_features/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Query data features linked to brain areas or spatial locations. These
    examples cover cellular, molecular, functional, connectivity, and other
    multimodal feature types.

  .. grid-item-card:: :material-outlined:`location_on;2em` Locations and regions of interest
    :link: examples/04_locations/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Define points, point sets, bounding boxes, and other spatial objects in
    reference coordinate systems. These examples show how locations are used
    in atlas workflows.

  .. grid-item-card:: :material-outlined:`push_pin;2em` Anatomical assignment
    :link: examples/05_anatomical_assignment/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Assign coordinates, regions of interest, and image maps to brain areas
    using labelled or statistical atlas maps.

  .. grid-item-card:: :material-outlined:`menu_book;2em` Extended tutorials
    :link: examples/tutorials/index.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    Work through longer examples that combine several siibra concepts in
    reproducible analysis workflows.


Legend
======

.. list-table::
   :header-rows: 1
   :widths: 20 15 60

   * - Label
     - Symbol
     - Meaning
   * - Basic
     - :bdg-primary:`Basic`
     - Short example introducing one main concept.
   * - Intermediate
     - :bdg-info:`Intermediate`
     - Example combining several concepts.
   * - Advanced
     - :bdg-warning:`Advanced`
     - Examples assuming familiarity with siibra concepts.
   * - Research workflow
     - :bdg-success:`Research workflow`
     - Notebook organized around a research-style question or analysis.
   * - Memory-heavy
     - :bdg-danger:`Memory-heavy`
     - May load larger objects or require more working memory.
   * - Network-heavy
     - :bdg-secondary:`Network-heavy`
     - May perform many remote requests and/or fetch larger data.


.. toctree::
  :hidden:
  :maxdepth: 3

  examples/01_atlases_and_parcellations/index
  examples/02_maps_and_templates/index
  examples/03_data_features/index
  examples/04_locations/index
  examples/05_anatomical_assignment/index
  examples/tutorials/index
