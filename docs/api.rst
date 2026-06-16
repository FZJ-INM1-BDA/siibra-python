.. -*- mode: rst -*-

.. _api:

=============
API reference
=============

This section contains the generated API reference for `siibra-python`. It
lists the main packages, modules, classes, and functions exposed by the Python
library.

The entries below are grouped by common tasks and by the main object types used
in siibra workflows. For workflow-oriented examples, see :ref:`examples`. For
the conceptual background, see :ref:`glossary`.

Class and package diagrams are available in the :ref:`developer documentation <developer>`.

Common entry points
===================

The following functions and classes are commonly used to start workflows or
construct core siibra objects.

.. list-table::
:header-rows: 1
:widths: 35 65

* * Task
  * Main entry points
* * Find a parcellation or brain area
  * :func:`siibra.get_parcellation`, :func:`siibra.get_region`
* * Find a reference coordinate system or template
  * :func:`siibra.get_space`, :func:`siibra.get_template`
* * Access atlas maps
  * :func:`siibra.get_map`
* * Define locations and regions of interest
  * :class:`siibra.Point`, :class:`siibra.PointSet`,
    :class:`siibra.BoundingBox`
* * Query multimodal data features
  * :func:`siibra.features.get`
* * Work with image data
  * :mod:`siibra.volumes`
* * Inspect dynamic feature sources
  * :mod:`siibra.livequeries`

.. autosummary::
:toctree: generated/

siibra.get_parcellation
siibra.get_region
siibra.get_space
siibra.get_template
siibra.get_map
siibra.Point
siibra.PointSet
siibra.BoundingBox
siibra.features.get

Main package
============

.. autopackagesummary:: siibra

Core atlas concepts
===================

Objects representing parcellations, brain areas, reference coordinate systems,
maps, and related atlas concepts.

.. autopackagesummary:: siibra.core

Locations and regions of interest
=================================

Objects for coordinates, point sets, bounding boxes, and spatial regions.

.. autopackagesummary:: siibra.locations

Volumes and image data
======================

Volume objects and providers for local, remote, and cloud-hosted image data.

.. autopackagesummary:: siibra.volumes

Data features
=============

Feature classes and query interfaces for multimodal measurements linked to
brain areas or spatial locations.

.. autopackagesummary:: siibra.features

Dynamic content from live queries
=================================

Interfaces that discover data features from external services at runtime.

.. autopackagesummary:: siibra.livequeries

Data retrieval and caching
==========================

Utilities for downloading, caching, and accessing external resources.

.. autopackagesummary:: siibra.retrieval

siibra-explorer integration
===========================

Utilities related to interaction with the siibra web viewer.

.. autopackagesummary:: siibra.explorer
