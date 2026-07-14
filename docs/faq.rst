.. _faq:

==========================
Frequently asked questions
==========================

This page provides short answers to common questions about
`siibra-python`. For a structured introduction to the terminology used
below, see :ref:`concepts`.

# Atlas concepts

.. dropdown:: What is siibra-python?

  `siibra-python` is the Python component of the siibra tool suite. It
  provides programmatic access to reference atlases, parcellations, reference
  spaces, maps, spatial locations, and linked multimodal data features.

  The default content implements the Multilevel Human Brain Atlas. The
  software can also work with local or project-specific atlas content.

  See :ref:`concepts` for an overview of the main atlas and data-access
  concepts.

.. dropdown:: What is the difference between an atlas, a parcellation, and a map?

  A **reference atlas** defines brain regions according to a particular
  organizational principle and provides spatial annotations of those regions.

  A **parcellation** represents the corresponding terminology of brain
  regions, often as a hierarchy.

  A **map** provides a spatial representation of the regions from a
  parcellation in a particular reference space.

  In other words, a parcellation describes which regions exist, while a map
  describes where those regions are represented.

  In the `siibra-python` API, an `Atlas` object groups compatible
  parcellations and reference spaces, usually for one species. See
  :ref:`concepts` for the distinction between this API object and a scientific
  reference atlas.

.. dropdown:: What is the difference between a labelled map and a statistical map?

  A **labelled map** assigns a discrete label to each represented spatial
  location. A label table connects each value to a brain region.

  A **statistical map** assigns continuous values to brain regions. Different
  regions may therefore overlap at the same location.

  A probabilistic map is a common type of statistical map. Its values may
  represent the frequency or estimated probability of finding a region at a
  location across a sample.

  See :ref:`concepts` and
  :doc:`examples/02_maps_and_templates/003_accessing_maps`.

.. dropdown:: What is a data feature?

  A **data feature** is a measurement or dataset linked to a brain region,
  spatial location, or another atlas concept.

  Features may describe cellular or molecular architecture, fibre
  architecture, function, connectivity, gene expression, electrophysiology,
  or macrostructural anatomy.

  Depending on the resource, feature data may be exposed as images, tables,
  numerical arrays, meshes, or links to external datasets.

  See :ref:`concepts` and
  :doc:`examples/03_data_features/000_matchings`.

.. dropdown:: What is an anatomical anchor?

  An anatomical anchor describes how a data feature is related to anatomy.

  A feature may be linked:

  * semantically, through a brain region or parcellation;
  * spatially, through a point, bounding box, image, or other location;
  * through a combination of semantic and spatial information.

  siibra evaluates these relationships when matching features to a query. See
  :ref:`concepts` for details.

# Data access and offline use

.. dropdown:: Does siibra-python download all atlas data during installation?

  No. Installing `siibra-python` does not download all atlas datasets.

  Atlas objects and metadata are loaded from configurations. Maps, templates,
  features, and other data are fetched lazily when a workflow requests them.

  This keeps the installation small and avoids downloading resources that are
  not used.

.. dropdown:: Why does the first query take longer than later queries?

  The first access to a map, template, or data feature may require retrieving
  metadata or data from an external resource.

  Retrieved files are stored in the local siibra cache and can usually be
  reused by later operations. Response times also depend on the requested
  resolution, data size, hosting service, and network connection.

.. dropdown:: Where are downloaded data stored?

  Retrieved files are stored in the local siibra cache.

  The cache avoids repeated downloads, but it has a configurable size limit.
  Files may be removed during cache maintenance when the limit is reached.

  See :doc:`offline_use` for cache preparation and configuration.

.. dropdown:: Can I prepare the cache before running a workflow?

  Yes. The most reliable approach is to run the complete workflow once while
  network access is available. This retrieves the resources that the workflow
  actually uses.

  `siibra.warm_cache()` can also preload registered objects and resources.
  Its default warm-up level prepares configured instances but does not
  necessarily download all referenced data. Data-level warm-up can be
  requested with `siibra.WarmupLevel.DATA`.

  Neither approach guarantees that every possible resource is available
  offline. In particular, complete preparation of large tiled image resources
  is not currently implemented.

  See :doc:`offline_use` for examples and limitations.

.. dropdown:: Can siibra-python be used without a network connection?

  Many workflows can run without network access when their configuration and
  required files have been prepared locally.

  However, some operations still require online services:

  * nonlinear transformations between reference spaces;
  * live queries to external data and metadata services;
  * access to uncached remote files;
  * requests for new regions or resolutions from tiled image resources.

  See :doc:`offline_use` for preparation instructions and a detailed overview
  of offline limitations.

.. dropdown:: Why are some image requests large?

  siibra provides access to resources ranging from small image files to
  terabyte-scale, multi-resolution image volumes.

  For large resources, restrict the requested spatial extent and resolution to
  what is needed by the workflow. A previously retrieved image region does not
  necessarily provide other regions or resolutions offline.

  See:

  * :doc:`examples/02_maps_and_templates/004_access_bigbrain`
  * :doc:`examples/tutorials/2025-paper-fig5`
  * :doc:`offline_use`

# Using your own data and atlas content

.. dropdown:: Can I use my own image or region of interest?

  Yes. Coordinates, point clouds, bounding boxes, image masks, and other
  spatial objects can be used in siibra workflows when their reference space
  is known.

  For example, an image-based region of interest can be assigned to brain
  regions or used to query spatially associated data features when the image
  is aligned with a supported reference space.

  See:

  * :doc:`use_your_own_data`
  * :doc:`examples/04_locations/000_employing_locations_of_interest`
  * :doc:`examples/05_anatomical_assignment/002_activation_maps`

.. dropdown:: Can I add my own atlas content or data features?

  Yes. A single custom object can be described in JSON and loaded with
  `siibra.from_json(...)`.

  For repeated use, related specifications can be collected in a local
  configuration that extends or replaces the default configuration.

  See :doc:`create_preconfiguration` for the complete workflow.

.. dropdown:: Do I need to modify siibra-python to add content?

  Usually not.

  Existing spaces, maps, volumes, and feature types can normally be added
  through JSON specifications. Code changes are only required when the content
  needs an unsupported object type, feature modality, provider, decoder, or
  live query.

  See :doc:`create_preconfiguration` and :ref:`developer`.

.. dropdown:: What is the difference between local content and a live query?

  Local content is explicitly supplied by the user, for example through a JSON
  specification or local configuration.

  A live query discovers or constructs features at runtime by communicating
  with an external service. Live queries are implemented in
  `siibra-python` and cannot operate without access to the corresponding
  service.

  See :ref:`concepts` and :doc:`offline_use`.

# Support, requests, and citation

.. dropdown:: Where should I ask usage questions?

  Public usage questions can be posted on
  `Neurostars <https://neurostars.org/tag/siibra>`_ using the `siibra` tag.

  Include, where possible:

  * a short description of the intended workflow;
  * a minimal code example;
  * the installed `siibra-python` version;
  * the complete error message or traceback.

.. dropdown:: Where should I report bugs?

  See :doc:`report_issues` for details.

.. dropdown:: Where should I request new functionality?

  See :doc:`report_issues` for details.

.. dropdown:: Where should I request new atlas content or data?

  Requests concerning parcellations, maps, spaces, templates, or configured
  data features should be discussed in the
  `siibra-configurations repository    <https://github.com/FZJ-INM1-BDA/siibra-configurations>`_.

  Include information about:

  * the scientific use case;
  * the proposed atlas element or dataset;
  * anatomical and spatial references;
  * publications or persistent identifiers;
  * stable data and metadata resources.

  See :doc:`create_preconfiguration` before preparing a contribution.

.. dropdown:: Where should I ask about EBRAINS accounts or infrastructure?

  Questions about EBRAINS accounts, permissions, hosted services, or
  infrastructure availability should be directed to the appropriate EBRAINS
  support channel rather than the `siibra-python` issue tracker.

.. dropdown:: How should I cite siibra-python?

  Cite the `siibra-python` software release used in the workflow.

  Where applicable, also cite the reference atlases, maps, data features, and
  source datasets retrieved through siibra.

  See :doc:`howtocite` for detailed citation guidance.

.. dropdown:: When should I clear the siibra cache?

Clear the cache after updating `siibra-python` if previously cached
resources may be incompatible with the new version. Clearing it is also a
useful troubleshooting step for incomplete, corrupted, or outdated cached
data.

.. code-block:: python

  import siibra

  siibra.cache.clear()

  This removes downloaded resources from the local siibra cache. Required
  files will be downloaded again when they are next accessed.

  Do not clear the cache immediately before working without network access,
  because the locally prepared resources will no longer be available.
