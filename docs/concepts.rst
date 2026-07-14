.. _concepts:

Main concepts
#############

The siibra tool suite integrates reference atlases, coordinate systems, brain
regions, spatial locations, and multimodal neuroscience data in a common
framework. Its components provide complementary interfaces:

* `siibra-explorer <https://atlases.ebrains.eu/viewer/>`_ supports interactive
  exploration in a web browser.
* `siibra-python <https://siibra-python.readthedocs.io/>`_ supports scripts,
  notebooks, analyses, and reproducible workflows.
* `siibra-api <https://siibra-api-stable.apps.hbp.eu/>`_ exposes siibra
  functionality through HTTP endpoints.

This page introduces the principal atlas concepts and the technical concepts
that determine how atlas content is represented and accessed. For details of
individual Python classes and functions, see :ref:`api`.

Atlas and anatomical concepts
==============================

Multilevel atlases
------------------

A **multilevel atlas** integrates several complementary reference atlases,
coordinate systems, and datasets across spatial scales and modalities.

It is therefore more than a single parcellation, template, or image. It provides
relationships that allow anatomical concepts, spatial locations, and
measurements to be interpreted together.

For example, the Multilevel Human Brain Atlas connects cytoarchitectonic brain
regions from Julich-Brain with macroscopic MNI templates, the microscopic
BigBrain template, and measurements such as receptor densities, cell
distributions, connectivity, and gene expression.

See also:

* :doc:`Anatomical characterization and multimodal profiling of regions of interest <examples/tutorials/2025-paper-fig3>`
* :doc:`Multimodal comparison of two cortical brain areas <examples/tutorials/2025-paper-fig4>`

Reference atlases and atlas objects
-----------------------------------

A **reference atlas** defines brain areas according to a particular
organizational principle and provides spatial annotations of those areas.

Examples include a cytoarchitectonic atlas, a fibre-bundle atlas, or an atlas of
functional modes. A multilevel atlas combines several such reference atlases.

The term `Atlas` has a more specific meaning in the `siibra-python` API. An
`Atlas` object mainly groups compatible parcellations and reference spaces,
usually for one species. The terminology and regions of an individual reference
atlas are normally represented by a :class:`~siibra.core.parcellation.Parcellation`
object and its associated maps.

For example, the Multilevel Human Atlas is available as an `Atlas` object,
while Julich-Brain is selected as one of its parcellations:

.. code-block:: python

  import siibra

  atlas = siibra.atlases.get("human")
  julich_brain = atlas.parcellations.get("julich")


See also:

* :doc:`Selecting a preconfigured atlas from an instance table <examples/01_atlases_and_parcellations/000_accessing_atlases>`
* :doc:`Selecting preconfigured parcellations <examples/01_atlases_and_parcellations/001_accessing_parcellations>`

Parcellation terminologies and brain regions
--------------------------------------------

A **parcellation** defines a terminology of brain regions. The terminology is
often hierarchical: broad anatomical structures contain progressively more
specific regions.

A **brain region** is a semantic anatomical concept. It is not identical to a
label value, voxel mask, or surface mesh. Those are spatial representations of
the region.

For example, `Area 44 (IFG) left` is a region in the Julich-Brain
terminology. It may be represented by different maps in different reference
spaces.

.. code-block:: python

  parcellation = siibra.parcellations.get("julich")
  region = siibra.get_region(parcellation, "Area 44 left")


See also:

* :doc:`Explore brain region hierarchies <examples/01_atlases_and_parcellations/002_explore_region_hierarchy>`
* :doc:`Find brain regions in a parcellation <examples/01_atlases_and_parcellations/003_find_regions>`
* :doc:`Basic brain region properties <examples/01_atlases_and_parcellations/004_brain_region_metadata>`
* :doc:`Spatial properties of brain regions <examples/01_atlases_and_parcellations/005_brain_region_spatialprops>`

Reference coordinate systems
------------------------------

A **reference coordinate system**, called a **space** in `siibra-python`,
defines how spatial positions in a brain are expressed.

Coordinates are only meaningful together with their reference space. The same
numerical coordinates can refer to different anatomical positions in different
spaces.

For example, `(27, -42, 63)` in the MNI 152 space does not describe the same
physical location as `(27, -42, 63)` in BigBrain space.

.. code-block:: python

  space = siibra.spaces.get("mni152")


See also:

* :doc:`Find predefined reference spaces <examples/02_maps_and_templates/001_selecting_reference_spaces>`

Reference templates
^^^^^^^^^^^^^^^^^^^

A **reference template** is a concrete spatial representation associated with a
reference coordinate system.

A template may be a volumetric image, a surface mesh, or another representation
of the reference anatomy. A single space can have templates with different
resolutions or data formats.

For example, the MNI 152 space provides an MRI-scale volumetric template,
whereas FreeSurfer spaces provide cortical surface representations.

.. code-block:: python

  space = siibra.spaces.get("mni152")
  template = space.get_template()


See also:

* :doc:`Accessing brain reference templates <examples/02_maps_and_templates/002_accessing_reference_templates>`
* :doc:`Access BigBrain high-resolution data <examples/02_maps_and_templates/004_access_bigbrain>`

Annotation sets (maps)
----------------------

An **annotation set** is a spatial realization of a brain-region terminology in
a particular reference coordinate system. It connects semantic brain regions to
voxels, vertices, or other spatial elements.

In `siibra-python`, annotation sets are accessed through map objects. A map
specifies:

* the parcellation whose regions it represents,
* the reference space in which it is defined,
* whether its representation is labelled or statistical,
* the available image or surface data.

For example, Julich-Brain has labelled and statistical representations in MNI
space:

.. code-block:: python

  labelled_map = siibra.get_map(
      parcellation="julich",
      space="mni152",
      maptype="labelled",
  )

  statistical_map = siibra.get_map(
      parcellation="julich",
      space="mni152",
      maptype="statistical",
  )

See also:

* :doc:`Accessing parcellation maps <examples/02_maps_and_templates/003_accessing_maps>`
* :doc:`Access parcellation maps in surface space <examples/02_maps_and_templates/005_surface_maps>`

Labelled maps
^^^^^^^^^^^^^

A **labelled map** assigns a discrete label to each represented spatial
location. A label table connects each label value to a brain region.

For example, a voxel value of `17` may identify a particular region. The
number itself has no anatomical meaning without the map's label table.

Labelled maps are useful for visualization, masking, and analyses in which each
location should be assigned to one region.

See also:

* :doc:`Accessing parcellation maps <examples/02_maps_and_templates/003_accessing_maps>`
* :doc:`Adding a custom parcellation map <examples/02_maps_and_templates/007_adding_custom_parcellation>`

Statistical and probabilistic maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **statistical map** assigns a continuous value to a brain region at each
represented location. Different regions may therefore overlap.

A **probabilistic map** is a statistical map whose values quantify the
estimated probability or frequency of finding a region at a location. Such maps
can represent anatomical variability across subjects.

For example, one coordinate may have a map value of `0.64` for Area 3b and
`0.18` for Area 1. The coordinate is therefore not forced into a single
discrete region.

See also:

* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`

Locations and regions of interest
-----------------------------------

A **location** describes a position or spatial extent in a reference coordinate
system. Supported location concepts include:

* points,
* point clouds,
* bounding boxes,
* image masks and image-based spatial objects.

A **region of interest** is a location selected for a particular analysis. It
may be an atlas region, coordinate, activation cluster, image mask, or bounding
box.

For example, an activation peak can be represented as a point, electrode
contacts as a point cloud, and a tissue sample as a bounding box.

.. code-block:: python

  point = siibra.Point(
      (27.75, -32.0, 63.725),
      space="mni152",
  )


See also:

* :doc:`Utilizing locations of interest <examples/04_locations/000_employing_locations_of_interest>`
* :doc:`Anatomical characterization and multimodal profiling of regions of interest <examples/tutorials/2025-paper-fig3>`

Location uncertainty
^^^^^^^^^^^^^^^^^^^^^

A location may have limited precision because of measurement resolution,
registration error, or uncertainty in a reported coordinate.

`siibra-python` can attach a spatial uncertainty to points. During
image-based assignment, an uncertain point can be represented by a Gaussian
volume instead of being treated as an exact mathematical position.

For example, an electrode position with an estimated uncertainty of
three millimetres can be represented as follows:

.. code-block:: python

  point = siibra.Point(
      (27.75, -32.0, 63.725),
      space="mni152",
      sigma_mm=3.0,
  )

This allows assignments to consider the surrounding volume and return overlap,
correlation, and containedness measures.

See also:

* :doc:`Utilizing locations of interest <examples/04_locations/000_employing_locations_of_interest>`
* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`

Spatial transformations
^^^^^^^^^^^^^^^^^^^^^^^

A **spatial transformation** relates locations in different reference
coordinate systems.

For example, a coordinate specified in MNI space can be transformed into
BigBrain space before it is compared with microscopic data:

.. code-block:: python

  point_mni = siibra.Point(
      (27.75, -32.0, 63.725),
      space="mni152",
  )
  point_bigbrain = point_mni.warp("bigbrain")


Spatial transformations are not exact. Their accuracy depends on the source and
target brains, the transformation method, local anatomical variability, and the
spatial scale of the analysis. Results near regional borders or at microscopic
resolution should therefore be interpreted carefully.

See also:

* :doc:`Utilizing locations of interest <examples/04_locations/000_employing_locations_of_interest>`
* :doc:`Anatomically guided reproducible extraction of full resolution image data
  from cloud resources <examples/tutorials/2025-paper-fig5>`

Anatomical assignment
^^^^^^^^^^^^^^^^^^^^^

**Anatomical assignment** relates a location or region of interest to brain
regions from a parcellation.

The comparison depends on the input object and map representation. Common
measures include:

* **incidence**, indicating whether a point lies inside a mapped region;
* **overlap**, quantifying the shared spatial extent of two objects;
* **containedness**, quantifying how much one object is contained in another;
* **correlation**, measuring spatial similarity between image-based objects.

For example, assigning an uncertain MNI coordinate to the Julich-Brain
statistical map can return several candidate regions with different map values,
correlations, and containedness scores.

.. code-block:: python

  parcellation_map = siibra.get_map(
      parcellation="julich",
      space="mni152",
      maptype="statistical",
  )

  assignments = parcellation_map.assign(point)

An assignment is therefore not necessarily a categorical decision. Multiple
results can preserve information about spatial overlap, anatomical variability,
and localization uncertainty.

See also:

* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`
* :doc:`Assign modes in activation maps to brain regions <examples/05_anatomical_assignment/002_activation_maps>`
* :doc:`Case study: Anatomical evaluation of subcortical maps <examples/tutorials/2025-paper-fig6>`

Data features
-------------

A **data feature** is a measurement or dataset linked to an anatomical concept
or spatial location.

Features may describe:

* cellular architecture,
* molecular architecture,
* fibre architecture,
* functional organization,
* structural or functional connectivity,
* gene expression,
* electrophysiology,
* macrostructural anatomy.

For example, receptor-density measurements can be queried for a
cytoarchitectonic region:

.. code-block:: python

  region = siibra.get_region("julich", "Area 44 left")

  features = siibra.features.get(
      region,
      siibra.features.molecular.ReceptorDensityFingerprint,
  )

See also:

* :doc:`Neurotransmitter receptor densities <examples/03_data_features/001_receptor_densities>`
* :doc:`Cortical cell body distributions <examples/03_data_features/003_cell_distributions>`
* :doc:`Gene expressions <examples/03_data_features/004_gene_expressions>`
* :doc:`Connectivity matrices <examples/03_data_features/006_connectivity_matrices>`

Anatomical anchors and match qualification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data feature may be linked to anatomy:

* **semantically**, through a brain region or parcellation;
* **spatially**, through a point, bounding box, image, or other location;
* through a combination of semantic and spatial information.

These links can have different levels of precision. `siibra-python` evaluates
the available relationships when matching a feature to a query.

For example, a receptor-density measurement may be linked directly to a
histologically identified brain region. A microscopic image may instead be
linked by a bounding box in BigBrain space. The first match is primarily
semantic, while the second is primarily spatial.

Information such as overlap, containedness, coordinate accuracy, and
parent-child relationships in a region hierarchy can contribute to the
qualification of a match.

See also:

* :doc:`Understanding links between data features and anatomical locations <examples/03_data_features/000_matchings>`

Software and data-access concepts
=================================

Separation of software and content
----------------------------------

siibra separates its software from its atlas content.

The software implements data models, queries, transformations, and data-access
mechanisms. Atlas elements and features are described separately and usually
refer to data hosted in external repositories or cloud services.

For example, a parcellation-map specification contains metadata, its reference
space, region associations, and pointers to the underlying image data. The
image itself does not need to be distributed with `siibra-python`.

This separation allows atlas content to evolve independently of software
releases.

See also:

* :doc:`Adding a custom parcellation map <examples/02_maps_and_templates/007_adding_custom_parcellation>`
* :doc:`Creating custom configurations <create_preconfiguration>`

Routes for providing content
----------------------------

siibra content can enter the system through three principal routes:

* foundational content,
* dynamic content,
* local content.

These routes differ in how content is discovered and maintained, but the
resulting objects can be used through the same main interfaces.

Foundational content
^^^^^^^^^^^^^^^^^^^^

**Foundational content** is defined by versioned JSON specifications collected
in a configuration.

A specification describes an atlas element or feature and points to its
metadata and data resources.

For example, a map specification may identify its parcellation, reference
space, map type, region labels, and image provider.

The default siibra configuration defines the foundational content of the
Multilevel Human Brain Atlas.

See also:

* :doc:`Creating custom configurations <create_preconfiguration>`

Dynamic content
^^^^^^^^^^^^^^^

**Dynamic content** is discovered at runtime through a **live query**.

A live query communicates with an external service and translates matching
resources into siibra feature objects. This avoids listing every available
feature in the static configuration.

For example, a live query can retrieve gene-expression measurements from an
external API or discover anatomically anchored datasets from a metadata
service.

From the user's perspective, dynamically discovered features are queried with
the same :func:`siibra.features.get` function as foundational features.

See also:

* :doc:`Gene expressions <examples/03_data_features/004_gene_expressions>`
* :doc:`EBRAINS regional datasets <examples/03_data_features/005_ebrains_datasets>`

Local content
^^^^^^^^^^^^^

**Local content** is supplied directly by a user and does not need to be part
of the default configuration.

For example, a local NIfTI image containing a custom parcellation can be
associated with its reference space and used in a local analysis.

See also:

* :doc:`Adding a custom parcellation map <examples/02_maps_and_templates/007_adding_custom_parcellation>`
* :doc:`Working with your data <use_your_own_data>`

Lazy data access and caching
----------------------------

Installing `siibra-python` does not download all atlas data.

Configuration metadata are loaded first. Image volumes, meshes, and feature
data are fetched only when a workflow requests them. Retrieved files are stored
in a configurable local cache and can be reused by later operations.

For example, selecting a reference template does not necessarily download its
image data. The data are retrieved when :meth:`~siibra.volumes.volume.Volume.fetch`
is called.

.. code-block:: python

  template = siibra.spaces.get("mni152").get_template()
  image = template.fetch()

The first access may therefore take longer than later accesses.

Configurations and required data can also be prepared locally for workflows
with restricted or unavailable network access.

See also:

* :doc:`Selecting a preconfigured atlas from an instance table <examples/01_atlases_and_parcellations/000_accessing_atlases>`
* :doc:`Access BigBrain high-resolution data <examples/02_maps_and_templates/004_access_bigbrain>`

Unified access to small and large images
----------------------------------------

siibra provides a common interface for conventional image files and very large
multi-resolution cloud images.

For a conventional image, fetching usually returns the original volume. For a
very large resource, siibra may return a suitable resolution or require a
region of interest that limits the amount of transferred data.

For example, a small cortical patch can be extracted from a terabyte-scale
BigBrain resource without downloading the complete brain volume. Cropping and
resampling are reflected in the spatial metadata of the resulting image, so the
patch remains localized in its original reference space.

See also:

* :doc:`Access BigBrain high-resolution data <examples/02_maps_and_templates/004_access_bigbrain>`
* :doc:`Anatomically guided reproducible extraction of full resolution image data
  from cloud resources <examples/tutorials/2025-paper-fig5>`

Efficient handling of statistical maps
--------------------------------------

Statistical parcellation maps may contain one volume for every represented
brain region. Loading all volumes as dense arrays can require substantial
download time and memory.

siibra can use a sparse representation that stores only non-zero regional
values. This is an internal optimization: assignment and map operations remain
available through the normal map interface.

For example, assigning a coordinate to a statistical Julich-Brain map can
evaluate all relevant regional values without requiring users to manage the
individual region volumes themselves.

See also:

* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`

Interoperable outputs and provenance
------------------------------------

siibra exposes fetched data through commonly used Python and neuroscience data
structures.

Typical representations include:

* NiBabel image objects for volumetric image data,
* pandas data frames for tabular measurements and assignment results,
* NumPy arrays for numerical data,
* mesh representations for surface data.

For example, fetching a volumetric map usually returns a NiBabel
`Nifti1Image`, while receptor-density measurements are exposed through
tabular data structures.

Atlas elements and features also retain metadata and links to their source
resources. These links make it possible to inspect provenance, publications,
dataset pages, and persistent identifiers where available.

See also:

* :doc:`Accessing parcellation maps <examples/02_maps_and_templates/003_accessing_maps>`
* :doc:`Neurotransmitter receptor densities <examples/03_data_features/001_receptor_densities>`
* :doc:`Understanding links between data features and anatomical locations <examples/03_data_features/000_matchings>`
