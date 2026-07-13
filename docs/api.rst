.. _api:

API reference
#############

This section contains the generated API reference for `siibra-python`. It
lists the main packages, modules, classes, and functions exposed by the Python
library.

The entries below are grouped by common tasks and by the main object types used
in siibra workflows. For workflow-oriented examples, see :ref:`examples`. For
the conceptual background, see :ref:`glossary`.

Class and package diagrams are available in the :ref:`developer documentation <developer>`.

Common entry points
===================

Query atlas elements
--------------------

The following functions and classes are commonly used to start workflows or
construct core siibra objects.

.. list-table::
  :header-rows: 1
  :widths: 35 65

  * * Task
    * Main entry points
  * * Find a parcellation
    * :func:`siibra.get_parcellation`
  * * Find an region/area
    * :func:`siibra.get_region`, :func:`siibra.find_regions`
  * * Find a reference coordinate system or template
    * :func:`siibra.get_space`, :func:`siibra.get_template`
  * * Access annotation sets
    * :func:`siibra.get_map`
  * * Query multimodal data features
    * :func:`siibra.features.get`

Define your regions of interest
-------------------------------

.. list-table::
  :header-rows: 1
  :widths: 35 65
  * * Define locations and regions of interest
    * :class:`siibra.Point`, :class:`siibra.PointCloud`,
      :class:`siibra.BoundingBox`
  * * Work with image/surface data
    * :func:`siibra.volumes.from_url`, :func:`siibra.volumes.from_file`,
    :func:`siibra.volumes.from_nifti`, :func:`siibra.volumes.from_pointcloud`,
    :func:`siibra.volumes.from_array`


Configuration and runtime settings
----------------------------------

These functions and objects control how siibra loads configuration data,
manages cached resources, adjusts runtime verbosity, and creates objects from
configuration dictionaries.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Purpose
   * - :func:`siibra.use_configuration`
     - Replace the active siibra configuration with a selected configuration source.
   * - :func:`siibra.extend_configuration`
     - Add a configuration source on top of the active default configuration.
   * - :data:`siibra.vocabularies`
     - Access controlled vocabularies used by siibra configuration and metadata.
   * - :data:`siibra.QUIET`
     - Reduce siibra's runtime output.
   * - :data:`siibra.VERBOSE`
     - Increase siibra's runtime output.
   * - :func:`siibra.warm_cache`
     - Pre-instantiate or fetch selected resources into the local cache.
   * - :func:`siibra.set_cache_size`
     - Set the maximum size of the local siibra cache.
   * - :func:`siibra.from_json`
     - Create a siibra object from a configuration dictionary.


Main package
============

.. autopackagesummary:: siibra

Core atlas concepts
-------------------

Objects representing parcellations, brain areas, reference coordinate systems,
maps, and related atlas concepts.

.. autopackagesummary:: siibra.core

Locations and regions of interest
---------------------------------

Objects for coordinates, point sets, bounding boxes, and spatial regions.

.. autopackagesummary:: siibra.locations

Volumes and image data
----------------------

Volume objects and providers for local, remote, and cloud-hosted image data.

.. autopackagesummary:: siibra.volumes

Data features
-------------

Feature classes and query interfaces for multimodal measurements linked to
brain areas or spatial locations.

.. autopackagesummary:: siibra.features

Data retrieval and caching
--------------------------

Utilities for downloading, caching, and accessing external or local resources.

.. autopackagesummary:: siibra.retrieval

siibra-explorer integration
===========================

Utilities related to interaction with the siibra web viewer.

.. autopackagesummary:: siibra.explorer
