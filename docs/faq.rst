.. _faq:

==========================
Frequently asked questions
==========================

This page collects short answers to common questions about using
`siibra-python`. For a structured explanation of the main terms used here,
see :ref:`concepts`.

General use
===========

.. dropdown:: What is siibra-python?

  `siibra-python` is the Python interface of the siibra tool suite. It
  provides programmatic access to reference atlases, brain-region
  terminologies, reference coordinate systems, maps of brain parcellations and
  regions, locations, and linked multimodal data features.

  The main content distributed with siibra implements the Multilevel Human
  Brain Atlas, but the software is designed to work with configurable atlas
  content.

.. dropdown:: What is the difference between a reference atlas, a parcellation, and a map?

  A reference atlas organizes anatomical knowledge. A parcellation defines a
  brain-region terminology, such as a set or hierarchy of named brain areas. A
  map is a spatial annotation of such a terminology in a reference coordinate
  system.

  In practice, a parcellation describes *what* the brain areas are, while a map
  describes *where* they are represented in a space.

  See :ref:`concepts` for a more complete explanation.

.. dropdown:: What is the difference between a labelled map and a statistical map?

  A labelled map assigns each spatial location to a discrete label. The label
  identifies a brain area from the corresponding terminology.

  A statistical map assigns continuous values to brain areas. A common special
  case is a probabilistic map, where the values express the probability of
  finding a brain area at a location. Statistical maps can represent overlap
  and uncertainty, which is important for atlases that capture inter-subject
  variability.

  See :ref:`concepts` for details.

.. dropdown:: What is a data feature?

  A data feature is a multimodal measurement or dataset linked to a brain area,
  spatial location, or other anatomical concept. Data features may include
  image data, tabular measurements, numerical arrays, or links to external
  dataset metadata.

  Data features can describe cellular architecture, molecular organization,
  fibre architecture, function, connectivity, or macrostructural anatomy.

Data access and performance
===========================

.. dropdown:: Does siibra-python download all atlas data during installation?

  No. `siibra-python` does not ship or download all atlas datasets during
  installation.

  The package loads metadata and object descriptions from configurations and
  retrieves data lazily. Image data, maps, templates, and data features are
  fetched only when a workflow requests them.

.. dropdown:: Why does the first query take longer than later queries?

  The first access to a map, template, or data feature may require downloading
  metadata or data from an external resource. Later accesses can reuse locally
  cached files when available.

  Response times can also depend on the size of the requested data, the
  selected resolution, the hosting service, and the network connection.

.. dropdown:: Where are downloaded data stored?

  Downloaded data are stored in the local siibra cache. The cache avoids
  repeated downloads and allows workflows to reuse data that have already been
  fetched.

  The exact cache location depends on the local environment and siibra
  configuration.

.. dropdown:: Can siibra-python be used offline?

  Typical first-time use requires network access because atlas content and data
  features point to distributed resources.

  Offline or restricted-network workflows require preparation, such as using a
  local or pre-populated cache and an appropriate configuration. See
  :doc:`create_preconfiguration` and :ref:`developer` for related
  implementation and configuration information.

.. dropdown:: Why are some data requests large?

  siibra can access data resources ranging from small files to large
  cloud-hosted image volumes. Some microscopic or high-resolution image
  resources are very large if requested at full resolution or across a large
  field of view.

  When working with large image resources, restrict the requested region,
  resolution, or feature selection to the part needed for the workflow.

Using own data
==============

.. dropdown:: Can I use my own image or region of interest with siibra-python?

  Yes. Local images, coordinates, bounding boxes, and other regions of interest
  can be used in siibra workflows when the required spatial reference
  information is available.

  For example, an image-based region of interest can be assigned to brain
  areas or used to query spatially linked data features, provided that the
  image is aligned to a supported reference coordinate system.

.. dropdown:: Can I add my own atlas content or data features?

  Atlas content and data features are generally not added by modifying the
  `siibra-python` source code directly. Foundational content is described in
  configuration files, while dynamic content can be discovered through live
  queries.

  See :doc:`create_preconfiguration` for information about defining and
  curating foundational content.

Support, requests, and citation
===============================

.. dropdown:: Where should I ask usage questions?

  Public usage questions and discussions can be posted on Neurostars using the
  `siibra` tag.

  Include a short description of the workflow, the relevant code snippet, and
  the installed `siibra-python` version when possible.

.. dropdown:: Where should I report bugs?

  Bugs should be reported on the GitHub issue tracker:

  ```
    https://github.com/FZJ-INM1-BDA/siibra-python/issues
  ```

  Include the installed `siibra-python` version, Python version, operating
  system, a minimal code example, and the complete error message or traceback.

.. dropdown:: Where should I request new features or data?

  Feature requests and data requests can be opened as GitHub issues. Please
  describe the requested behavior or dataset, the intended use case, and any
  relevant references or persistent identifiers.

  For questions about EBRAINS services, account access, or infrastructure
  availability, use the EBRAINS support channels.

.. dropdown:: How should I cite siibra-python?

  Cite the `siibra-python` software release used in the workflow and, where
  applicable, cite the atlas content, data features, and datasets retrieved
  through siibra.

  See :doc:`howtocite` for citation details.
