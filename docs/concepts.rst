.. _glossary:

Glossary
========

siibra provides programmatic access to brain atlases and linked multimodal
data resources. It connects anatomical concepts, spatial reference systems,
atlas maps, locations in the brain, and data features in a common framework.

This page introduces the main concepts used throughout the siibra
documentation. It is intended as a user-facing overview. For the complete
Python API, see :ref:`api`.

What siibra connects
--------------------

A brain atlas is more than a single image or parcellation file. In siibra,
atlas content is represented as a set of connected concepts:

* **reference atlases**, which organize anatomical knowledge,
* **terminologies**, which define named brain areas,
* **reference coordinate systems**, also called **spaces**, in which locations
  can be expressed,
* **reference templates**, which represent anatomy in a space,
* **annotation sets**, which map brain areas into a space,
* **locations** and **regions of interest**, which describe user-defined
  positions or spatial extents in the brain,
* **data features**, which provide multimodal measurements linked to brain
  areas or locations.

The main content distributed with siibra implements the Multilevel Human Brain
Atlas. It links reference atlases and coordinate systems across spatial scales,
from macroscopic MRI templates and cortical surfaces to microscopic BigBrain
resources, and connects them with multimodal data features.

siibra separates software from content. The Python package provides the
interfaces, data models, and access mechanisms. The actual atlas content and
data features are described in configuration files or discovered through live
queries, and point to data hosted in external repositories.

Reference atlases
-----------------

A **reference atlas** is a structured anatomical resource. It provides a set of
brain areas and one or more spatial representations of these areas.

A reference atlas should not be confused with a single parcellation image. A
single atlas may contain several maps in different spaces or with different
representation types. For example, an atlas may provide probabilistic maps in
an MNI space and another representation in a surface or microscopic space.

In current user-facing siibra-python workflows, atlas content is usually
accessed directly through functions such as :func:`siibra.get_parcellation`,
:func:`siibra.get_region`, :func:`siibra.get_space`, and
:func:`siibra.get_map`. Atlas objects remain available for inspecting
atlas-level metadata.

Terminologies and brain areas
-----------------------------

A **terminology** defines the anatomical concepts used by an atlas. In many
cases, a terminology is hierarchical: large anatomical structures contain
smaller substructures, and named brain areas may have aliases or alternative
names.

A **brain area** is therefore a semantic concept, not just a number in an image.
The same brain area may have different spatial representations in different
maps, spaces, or spatial resolutions.

In the siibra-python API, the term `region` is used in many places for such
brain-area concepts. In the documentation, we use **brain area** when referring
to the anatomical concept, and `region` when referring to the corresponding
API argument or object.

Reference coordinate systems and templates
------------------------------------------

A **reference coordinate system** defines how locations in the brain are
specified. In siibra, this is often called a **space**.

A **reference template** is a spatial representation of anatomy in such a
space. Depending on the space, a template may be a 3D image volume, a surface
mesh, or another spatial representation.

Examples of human reference spaces used by siibra include:

* the MNI ICBM 152 2009c nonlinear asymmetric space,
* the MNI Colin27 space,
* the BigBrain space,
* FreeSurfer surface spaces such as fsaverage and fsaverage6.

These spaces differ in scale, subject basis, topology, and typical use case.
MNI spaces are commonly used for MRI-scale neuroimaging analyses. BigBrain is
a microscopic 3D reconstruction of a single postmortem human brain. Surface
spaces represent cortical anatomy on meshes rather than in voxel volumes.

Annotation sets, maps, and parcellations
----------------------------------------

An **annotation set** is a spatial representation of a terminology in a
reference coordinate system. It assigns brain-area information to locations in a
space.

In siibra-python, annotation sets are usually accessed as maps, for example
with :func:`siibra.get_map`.

siibra distinguishes different map types.

**Labelled maps**

A **labelled map** assigns each spatial location to a discrete label. The label
identifies a brain area from the corresponding terminology.

Labelled maps are useful when each location should belong to one area only.
They are often compact and convenient for visualization, masking, and
region-wise analyses.

**Statistical and probabilistic maps**

A **statistical map** assigns continuous weights to brain areas. A common
special case is a **probabilistic map**, where the weights express the
probability of finding a brain area at a location.

In a statistical map, different brain areas may overlap. A single coordinate
can therefore be associated with multiple brain areas, each with a different
weight.

This is important for atlases that represent inter-subject variability, such as
probabilistic cytoarchitectonic maps. Rather than forcing each coordinate into
a single label, statistical maps preserve uncertainty and overlap between brain
areas.

**Parcellation maps**

A **parcellation map** is a map that subdivides a brain or brain structure into
a set of regions. In practice, the term is often used for whole-brain discrete
label maps. In this documentation, we use **parcellation map** when referring
to such a spatial subdivision, and **map** as the more general term covering
labelled, statistical, volumetric, and surface-based representations.

Locations and regions of interest
---------------------------------

A **location** describes a position or spatial extent in a reference coordinate
system. siibra supports different forms of locations, including:

* points,
* point sets,
* bounding boxes,
* image masks,
* image maps,
* brain areas used as semantic locations.

A **region of interest** (ROI) is a user-defined location or spatial object that
is used for an analysis task. An ROI may be a coordinate, a cluster from an
activation map, a mask image, a bounding box, or a brain area selected from an
atlas.

Every spatial location in siibra belongs to a reference coordinate system. This
allows siibra to compare locations, maps, and data features safely, and to apply
coordinate transformations when needed.

**Uncertain locations**

Some locations are not exact. For example, a reported activation coordinate
may have limited spatial precision, or a measurement may correspond to a small
volume rather than to a mathematical point.

siibra can represent such cases using spatial objects rather than only point
coordinates. In some workflows, uncertain coordinates can be treated as small
image regions, allowing assignment and feature queries to account for spatial
uncertainty.

Spatial transformations
-----------------------

Many workflows require comparing information from different reference
coordinate systems. For example, a user-defined coordinate may be specified in
an MNI space, while a relevant microscopic image feature may be anchored in
BigBrain space.

siibra can use precomputed spatial transformations to relate coordinates across
supported spaces. These transformations make it possible to preserve locations
when switching spaces, query data features across spaces, or compare maps that
are available in different representations.

Spatial transformations are useful, but they are not exact. Their precision
depends on the source and target spaces, the transformation model, the local
anatomy, and the spatial scale of the analysis. When interpreting results,
especially at microscopic resolution or near areal borders, the uncertainty
introduced by cross-space transformations should be considered.

Anatomical assignment
---------------------

**Anatomical assignment** links a user-defined location or ROI to brain areas
from a reference atlas.

Depending on the input and map type, assignment may use different measures:

* **incidence**, for example whether a point falls inside a labelled region,
* **overlap**, for example how much an ROI overlaps with a brain-area map,
* **correlation**, for example how similar an image map is to a statistical
  brain-area map.

For statistical and probabilistic maps, assignment can return multiple
candidate brain areas with weights, probabilities, overlap values, or
correlation scores. This is often preferable to assigning a location to only one
area, because it preserves uncertainty and inter-subject variability.

Anatomical assignment is a central operation in siibra. It allows coordinates,
masks, activation clusters, and other ROIs to be characterized in terms of known
brain areas.

Data features
-------------

A **data feature** is a multimodal measurement or dataset linked to an
anatomical concept or spatial location. Data features can describe different
aspects of brain organization, such as:

* cellular architecture,
* molecular organization,
* fibre architecture,
* functional organization,
* structural or functional connectivity,
* macrostructural anatomy.

Data features may be linked semantically to brain areas, spatially to
coordinates or bounding boxes, or by a combination of anatomical and spatial
metadata.

In siibra-python, data features are commonly queried with
:func:`siibra.features.get`. A query can start from a brain area, a map, a
location, or another supported concept.

Data features can expose different data structures depending on the underlying
resource and modality. Common examples include:

* image data, represented as NiBabel image objects where possible,
* tabular data, represented as pandas DataFrames,
* numerical arrays, represented as NumPy arrays,
* metadata links to external dataset pages.

This makes data retrieved through siibra compatible with common Python tools
for neuroimaging, statistics, visualization, and scientific computing.

Content, configurations, and live queries
-----------------------------------------

siibra content can enter the system in different ways.

**Foundational content** is defined by configuration files. These files describe
atlas elements such as spaces, templates, maps, brain areas, and selected data
features. They contain metadata and references to external resources, but not
necessarily the data itself.

The default configuration used by siibra-python provides the foundational
content of the Multilevel Human Brain Atlas.

**Dynamic content from live queries** is discovered at runtime by querying
external services. Live queries allow siibra to retrieve or construct data
feature descriptions from supported resources without requiring every feature to
be listed statically in the configuration.

For example, live queries can connect siibra to external metadata services,
image resources, or modality-specific APIs, and expose the results through the
same feature mechanisms used for foundational content.

Local user content
~~~~~~~~~~~~~~~~~~

Local or custom content can also be used in siibra workflows. Examples include
a local image, a custom region of interest, or a custom map for a specific
analysis.

Such content can be combined with siibra objects as long as the required
spatial reference information is available.

Data access and caching
-----------------------

siibra uses lazy data access. Metadata and object descriptions can be inspected
without downloading all associated data. Actual data are fetched only when they
are requested.

Fetched data may be stored in a local cache. This avoids repeated downloads and
allows workflows to become faster after the first access.

siibra provides a common user-level interface for different kinds of image
resources. Small file-based images and large cloud-hosted image volumes can be
accessed through similar volume objects. When only a region or lower resolution
is requested from a large image resource, siibra fetches the requested part and
keeps the spatial metadata consistent with the original reference space.

How these concepts appear in siibra-python
------------------------------------------

The following API entry points are commonly used when working with the
concepts described above:

``siibra.get_atlas(...)``
    Find a reference atlas. This is useful for inspecting atlas metadata or
    explicitly selecting an atlas context.

``siibra.get_region(...)``
    Find a brain area from a terminology.

``siibra.get_parcellation(...)``
    Find a brain-region terminology, such as Julich-Brain.

``siibra.get_space(...)``
    Find a reference coordinate system.

``siibra.get_map(...)``
    Find and access an annotation map for a given parcellation, space, and map
    type.

``space.get_template()``
    Quick access to a reference template for a coordinate system.

``siibra.Point``, ``siibra.PointCloud``, ``siibra.BoundingBox``
    Define spatial locations in a reference coordinate system.

``siibra.features.get(...)``
    Query data features linked to a brain area, location, map, or other
    supported concept.

Together, these concepts allow siibra workflows to move between anatomical
names, spatial coordinates, atlas maps, and multimodal data while preserving
the links between them.


References
----------

.. [#Fonov2011] Fonov V., Evans A.C., Botteron K., Almli C.R.,
   McKinstry R.C., Collins D.L. Unbiased average age-appropriate atlases for
   pediatric studies. NeuroImage, 54(1), 313-327, 2011.
   doi: 10.1016/j.neuroimage.2010.07.033

.. [#Dale1999] Dale A.M., Fischl B., Sereno M.I. Cortical Surface-Based
   Analysis: I. Segmentation and Surface Reconstruction. NeuroImage, 9(2),
   179-194, 1999. doi: 10.1006/nimg.1998.0395

.. [#Fischl1999] Fischl B., Sereno M.I., Dale A.M. Cortical Surface-Based
   Analysis: II: Inflation, Flattening, and a Surface-Based Coordinate System.
   NeuroImage, 9(2), 195-207, 1999. doi: 10.1006/nimg.1998.0396

.. [#BigBrain2013] Amunts K., Lepage C., Borgeat L., Mohlberg H.,
   Dickscheid T., Rousseau M.-E., Bludau S., Bazin P.-L., Lewis L.B.,
   Oros-Peusquens A.-M., Shah N.J., Lippert T., Zilles K., Evans A.C.
   BigBrain: An Ultrahigh-Resolution 3D Human Brain Model. Science,
   340(6139), 1472-1475, 2013. doi: 10.1126/science.1235381

.. [#JulichBrain2020] Amunts K., Mohlberg H., Bludau S., Zilles K.
   Julich-Brain: A 3D probabilistic atlas of the human brain's
   cytoarchitecture. Science, 369(6506), 988-992, 2020.
   doi: 10.1126/science.abb4588

.. [#Wagstyl2020] Wagstyl K. et al. BigBrain 3D atlas of cortical layers:
   Cortical and laminar thickness gradients diverge in sensory and motor
   cortices. PLoS Biology, 18(4), e3000678, 2020.
   doi: 10.1371/journal.pbio.3000678

.. [#AtOM2023] Kleven H., Gillespie T.H., Zehl L., Dickscheid T.,
   Bjaalie J.G., Martone M.E., Leergaard T.B. AtOM, an ontology model to
   standardize use of brain atlases in tools, workflows, and data
   infrastructures. Scientific Data, 10, 486, 2023.
   doi: 10.1038/s41597-023-02389-4

.. [#Lebenberg2018] Lebenberg J. et al. A framework based on sulcal
   constraints to align preterm, infant and adult human brain images acquired
   in vivo and post mortem. Brain Structure and Function, 223(9),
   4153-4168, 2018. doi: 10.1007/s00429-018-1735-9
