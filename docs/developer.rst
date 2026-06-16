.. _developer:

=======================
Developer documentation
=======================

This page describes implementation concepts used in `siibra-python`. It is
intended for contributors who work on the package internals, add new object
types, or maintain interfaces between atlas concepts, data access, and feature
queries.

For the generated API reference, see :ref:`api`. For the user-facing concept
guide, see :ref:`glossary`. For adding or curating foundational content, see
:doc:`create_preconfiguration`.

Architecture overview
=====================

`siibra-python` is the core implementation layer of the siibra tool suite. It
models atlas concepts, spatial locations, maps, volumes, data features, live
queries, and retrieval mechanisms. The web API and viewer build on the same
conceptual model, but this page focuses on the Python package.

A central design principle is the separation of code from content. The package
implements data models and access mechanisms, while atlas contents are defined
by configuration files, discovered through live queries, or provided locally in a
workflow. Actual image, tabular, and numerical data are fetched lazily from
local files or external resources.

Package overview
================

The following diagram gives a high-level view of the main packages and their
roles. It is intentionally less detailed than automatically generated package
diagrams, and focuses on the relationships that are most relevant for
development.

.. mermaid::

  flowchart TB
  siibra["siibra<br/>public entry points"]

  ```
    core["siibra.core<br/>atlas concepts"]
    locations["siibra.locations<br/>points, point sets, bounding boxes"]
    volumes["siibra.volumes<br/>maps, volumes, providers"]
    features["siibra.features<br/>multimodal data features"]
    livequeries["siibra.livequeries<br/>dynamic content"]
    retrieval["siibra.retrieval<br/>requests, cache, datasets"]
    configuration["siibra.configuration<br/>foundational content"]
    explorer["siibra.explorer<br/>viewer integration"]

    siibra --> core
    siibra --> locations
    siibra --> volumes
    siibra --> features

    core --> configuration
    volumes --> retrieval
    features --> retrieval
    features --> livequeries
    livequeries --> retrieval
    core --> locations
    core --> volumes
    features --> core
    features --> locations
    explorer --> core
    explorer --> locations
  ```

The most important implementation boundary is between object metadata and data
retrieval. Objects such as parcellations, spaces, maps, regions, and features
can usually be instantiated without loading their underlying image or tabular
data. Data transfer is deferred until methods such as `fetch()` are called.

Core class relationships
========================

The following diagram shows the central class relationships in simplified form.
It is not a complete inheritance diagram. The generated API reference remains
the source for detailed class and method signatures.

.. mermaid::

  classDiagram
    class AtlasConcept {
    +id
    +name
    +key
    +registry()
    +get_instance(spec)
    }

    ```
      class Parcellation {
        +get_region(regionspec)
        +get_map(space, maptype, spec)
      }

      class Region {
        +find(regionspec)
        +supported_spaces
      }

      class Space {
        +get_template()
      }

      class Map {
        +space
        +parcellation
        +maptype
        +get_volume(region, index)
        +fetch(region, index, **kwargs)
        +fetch_iter(**kwargs)
        +compress(**kwargs)
      }

      class SparseMap {
        +sparse_index
        +affine
        +shape
      }

      class SparseIndex {
        +voxels
        +probs
        +bboxes
        +load(filepath_or_url)
        +save(base_filename, folder)
        +from_sparsemap(sparsemap)
      }

      class Volume {
        +providers
        +fetch(**kwargs)
      }

      class VolumeProvider {
        +fetch(**kwargs)
      }

      class Location
      class Point
      class PointSet
      class BoundingBox

      class Feature {
        +get_instances()
        +anchor
        +fetch()
      }

      class LiveQuery {
        +FeatureType
      }

      AtlasConcept <|-- Parcellation
      AtlasConcept <|-- Region
      AtlasConcept <|-- Space
      AtlasConcept <|-- Map

      Region <|-- Parcellation
      Map <|-- SparseMap

      SparseMap *-- SparseIndex
      Map *-- Volume
      Volume *-- VolumeProvider

      Location <|-- Point
      Location <|-- PointSet
      Location <|-- BoundingBox

      Feature --> AtlasConcept
      Feature --> Location
      LiveQuery --> Feature
    ```

`AtlasConcept` implements common behavior for configured atlas concepts,
including identifiers, names, keys, matching, and registries. `Parcellation`
represents a brain-region terminology and extends `Region` because a
parcellation is also the root of a region hierarchy. `Map` represents a
spatial annotation set for a parcellation in a reference coordinate system.
`SparseMap` specializes `Map` for sparse statistical maps.

Object instantiation
====================

siibra creates many objects during normal use, but most objects are lightweight
descriptions until their data are explicitly requested. Object creation follows
two main routes.

 Foundational content from configurations
-----------------------------------------

Foundational objects are built from configuration files. This applies to
subclasses of `AtlasConcept` whose class parameter `configuration_folder`
points to a folder in the active configuration, and to subclasses of `Feature`
that are described by feature specifications.

For `AtlasConcept` subclasses, instance construction is centralized in
`AtlasConcept.registry()` and `AtlasConcept.get_instance()`. Subclasses are
registered through `AtlasConcept.__init_subclass__()`, and their runtime
registries are stored as `InstanceTable` objects.

For feature subclasses, configured feature instances are collected through
`Feature.get_instances()`. Feature instances are stored at class level rather
than in `InstanceTable` registries.

 Dynamic content from live queries
----------------------------------

Dynamic content from live queries is provided by subclasses of `LiveQuery`.
A live query declares the feature class it produces through its `FeatureType`
class parameter. During subclass creation, the live query is registered for the
corresponding feature type.

When `Feature.get_instances()` is called, registered live queries can extend
the configured feature list with feature instances discovered from external
services at runtime.

Design patterns
===============

 Avoiding cyclic imports
------------------------

The package contains many interacting modules. Importing concrete classes from
neighboring modules at file import time can create cyclic import dependencies.

Prefer importing local modules and accessing classes through the module
namespace in implementation files.

.. code-block:: python
  :caption: Prefer module imports in implementation files

  from ..core import space

  def my_func():
  s = space.Space.get_instance("mni152")
  return s

Avoid direct local class imports in implementation files when they are not
needed at import time.

.. code-block:: python
  :caption: Avoid direct local class imports when they create cycles

  from ..core.space import Space

  def my_func():
  s = Space.get_instance("mni152")
  return s

Direct imports of selected classes are typically most appropriate in
`__init__.py` files, where the public package interface is assembled.

 Lazy loading via properties
----------------------------

Many objects are preconfigured but never used in a given workflow. Expensive
operations should therefore be delayed until the corresponding object property
or method is accessed.

A common pattern is to keep a private cache attribute and fill it on first use.

.. code-block:: python
  :caption: Lazy loading pattern

  class Thing:
  def **init**(self):
  self._heavy_property_cached = None

  ```
    @property
    def heavy_property(self):
        if self._heavy_property_cached is None:
            self._heavy_property_cached = some_heavy_computation()
        return self._heavy_property_cached
  ```

This pattern is used for metadata-derived relationships, registries, sparse
indices, image properties, and other objects whose construction may require
network access, disk access, or substantial computation.

Handling volumes
================

A `Volume` represents a spatial data object, such as an anatomical template,
an annotation map, or another image or mesh resource. A volume may describe a
complete 3D object, a surface representation, or a resource that can be fetched
at different resolutions or formats.

Important terms:

`Volume`
The object that represents image or mesh data at the siibra level.

`VolumeProvider`
The object that knows how to retrieve a particular representation of a
volume from a file, URL, cloud image resource, or in-memory object.

`Variant`
An alternative representation of the same volume, for example an inflated
surface.

`Fragment`
An individually addressable part of a volume, for example a left or right
hemisphere.

`MapIndex`
A mapping from a brain area to a volume index, label, fragment, or
subvolume specification.

`z`
A fourth-axis coordinate used when several 3D maps are stored in a 4D
image volume.

`format`
The requested data format, such as an image or mesh format supported by the
available providers.

Fetching data is explicit. A template, map, or feature can describe available
data without immediately loading it. Calling `fetch()` selects an appropriate
provider, retrieves the data, and returns it in a Python-compatible form. Image
data are typically returned as NiBabel image objects, while meshes are typically
returned as dictionaries with vertices, faces, and optionally labels.

The volume-fetching flow can be summarized as follows:

.. mermaid::

  flowchart LR
  request["Template, map, or feature"]
  volume["Volume"]
  provider["VolumeProvider"]
  data["Fetched image or mesh"]

  ```
    request -->|"selects volume"| volume
    volume -->|"selects provider<br/>by format, variant, fragment, resolution, VOI"| provider
    provider -->|"loads data"| data
  ```

For maps, `Map.fetch()` first resolves the relevant `Volume` or
`FilteredVolume` using the requested region or `MapIndex`. The resulting
volume then delegates retrieval to its provider.

Sparse statistical maps
=======================

Statistical maps, including probabilistic maps, may contain one 3D volume per
brain area. For large parcellations, loading all regional maps as dense volumes
can require substantial memory although most voxels are zero.

`SparseMap` provides a sparse representation for such maps. It subclasses
`Map` and uses a `SparseIndex` to store only non-zero region assignments.

The sparse representation consists of three main structures:

`SparseIndex.voxels`
A 3D integer image in the map voxel space. Values below zero indicate that
no region has a non-zero value at that voxel. Non-negative values are
indices into `SparseIndex.probs`.

`SparseIndex.probs`
A list of dictionaries. Each dictionary stores the non-zero map values for
one occupied voxel. Keys are map indices, and values are statistical or
probabilistic weights.

`SparseIndex.bboxes`
A list of per-volume bounding boxes. These allow assignment and lookup
operations to restrict work to the spatial extent of each regional map.

The relation between these structures is illustrated below.

.. mermaid::

  flowchart TB
  dense["Dense statistical maps<br/>one 3D map per brain area"]
  sparsemap["SparseMap<br/>Map subclass for statistical maps"]
  sparseindex["SparseIndex"]

  ```
    voxels["voxels<br/>3D integer array<br/>-1: no assignment<br/>0..N: index into probs"]
    probs["probs<br/>list of dictionaries<br/>probs[i] = {map_index: value}"]
    bboxes["bboxes<br/>per-map bounding boxes"]

    query["Lookup at voxel (x, y, z)"]
    i["i = voxels[x, y, z]"]
    empty["i < 0<br/>no assigned region"]
    assigned["i >= 0<br/>read probs[i]"]

    dense -->|"converted or loaded as"| sparsemap
    sparsemap --> sparseindex
    sparseindex --> voxels
    sparseindex --> probs
    sparseindex --> bboxes

    query --> i
    i --> empty
    i --> assigned
    assigned --> probs
  ```

For example, if `SparseIndex.voxels[x, y, z]` returns `42`, the region
weights at this voxel are stored in `SparseIndex.probs[42]`. If the value is
negative, no statistical map has a non-zero value at this voxel.

`SparseIndex` can be built from a `SparseMap` with
`SparseIndex.from_sparsemap()`, loaded with `SparseIndex.load()`, and saved
with `SparseIndex.save()`. The serialized representation uses three files:
one for voxel indices, one for probability dictionaries, and one for bounding
boxes.

`SparseMap.sparse_index` resolves the sparse index lazily. It first attempts
to load a cached index, then a precomputed index, and finally computes the
index from the map volumes if necessary. The computed index is saved to the
local siibra cache.

Multimodal data features
========================

Feature classes represent multimodal measurements linked to atlas concepts or
spatial locations. Feature instances may originate from configuration files,
dynamic content from live queries, or local workflow objects.

Each feature has an anatomical anchor. The anchor describes the semantic or
spatial relationship between the feature and the atlas. Examples include a
parcellation, a brain area, a location, or a combination of brain area and
location.

Feature queries use these anchors to determine whether a feature is relevant
for a requested concept. This allows the same query interface to work with
features linked by brain area, coordinate space, bounding box, or other
supported anatomical metadata.

Adding or curating foundational feature specifications is covered in
:doc:`create_preconfiguration`.
