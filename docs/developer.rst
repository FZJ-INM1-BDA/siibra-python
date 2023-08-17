=======================
Developer documentation
=======================


I. Object instantiation
=======================

Since siibra creates many objects in a typical workflow, the creation of objects is following a clear structure.
There are two main approaches of object creation:

1. Preconfigured objects, via the configuration module. This happens automatically
   for
    * subclasses of `AtlasConcept`, if the class parameter `configuration_folder`
      points to a subfolder of siibra configuration repositories.
    * subclasses of `Feature`, if the class parameter `configuration_folder` points
      to a subfolder of siibra configuration repositories.
   The implementation of these queries is centralized in `AtlasConcept.get_instances()`
   and `Feature.get_instances()`, respectively, and relies on the registrations
   done in `AtlasConcept.__init_subclass__()` and `Feature.__init_subclass__()`.

2. Live queries, via the livequery module. This applies to subclasses of `Feature`,
   and relies on subclasses of `LiveQuery` with the class parameter `FeatureType`
   set to the corresponding subclass of `Feature`. This triggers the query to be
   registered in the `Feature._live_queries` list via `LiveQuery.__init_subclass__()`.
   Any registered live queries will then automatically be called in
   `Feature.get_instances()`, extending the list of possible preconfigured
   object instances.

Note that `AtlasConcept` keeps the list of instances in an `InstanceTable` during
runtime, which can be accessed via `AtlasConcept.registry()` for any subclass.
Instances of `Feature` subclasses are just stored as a list at class level.


II. Frequently used design patterns
===================================

`siibra` makes frequent use of some typical design patterns.


Importing local classes
-----------------------

Often, siibra implementations make use of classes from different siibra modules.
It is common Python practice to do all imports at the start of a file, and not
locally inside functions / classes.

Direct imports of classes from other local modules like here:

.. code-block:: python
    :caption: Imports of local classes (should be avoided)

    from ..core.space import Space

    def my_func():
        s = Space()


result in immediate creation of the class, and quickly leads to cyclic import
dependencies which result in runtime errors.

This effect can often be avoided by only importing the siibra module, and
deferring the class creating to a later stage in the code:

.. code-block:: python
    :caption: Imports of local modules

    from ..core import space

    def my_func():
        s = space.Space()

In general, it seems a good practice to import specific classes only in the
`__init__.py` files, and use module imports in other python files.
However, this rule of thumb is not yet consistently implemented and verified in siibra. 


Lazy loading via object properties
----------------------------------

Since siibra pre-configures many objects, of which the user will typically only
use a few  (e.g. after filternig data features by brain regions), it is
important that time and/or memory consuming operations are only executed when
objects are actually requested and used.  We typically solve this by implementing
object properties with a lazy loading mechanism, following this scheme:

..  code-block:: python
    :caption: Lazy loading principle

    class Thing:
        def __init__(self):
            self._heavy_property_cached = None

        @property
        def heavy_property(self):
            if self._heavy_property_cached is None:
                # only here we do the initialization,
                # and only once for the object
                self._heavy_property_cached = some_heavy_computation()
            return self._heavy_property_cached


III. Handling Volumes
=====================

Basic definitions and notes
---------------------------

* **Volume:** is a complete 3D object, typically a complete brain.
* **Volume provider:** is a resource that provides access to volumes. A volume
  can have multiple providers in different formats.
* **Variant:** refers to alternative representations of the same volume.
  (e.g. inflated surface).
    * If the volume has variants, they need to be listed in the configuration file.
* **Fragments:** are individually addressable components of a volume.

    * If a volume has fragments, either the user or the code needs to retrieve
      from multiple sources to access the complete volume.
    * Fragments need to be named (e.g. left and right hemisphere), because they
      inevitably split the whole object into distinct anatomical parts that
      require semantic labeling.
* **Brain regions (label):** are structures mapped inside a specific volume or fragment.

    * The structure appears by interpreting the labels inside the volume listed in
      the configuration file.
        * In special cases, a brain region could be represented by the complete
          volume or fragment.
* **Volume index:** the index of the volume in case there is more than one;
  typically used for probability maps, where each area has a different volume.
* **Z:** for 4D volumes, it specifies the 4th coordinate identifying an actual
  3D volume. It has a similar function as the volume index, only that the volumes
  are concatenated in one array and share the same affine transformation.
* **Source type (format):** the format of the volume data.

    * See :data:`SUPPORTED_FORMATS` (:data:`IMAGE_FORMATS` and :data:`SURFACE_FORMATS`)
      at volumes.volume.py for the currently supported formats.

Fetching volumes
----------------

Fetching volumes occurs in two main stages:

1. The determination of the volume by the user.
   
  * The user sets the object they would like to fetch a volume from:

     * a space template -> using `get_template()` which provides a volume template.
     * or a map -> getting the desired map by setting desired specs.
  
  * The user invokes `fetch()` method to retrieve the volume from the template or map.

     * template directly accesses to `volume.fetch()`
     * `fetch()` first goes through `map.fetch()` to determine the associated volume.

2. Actual retrieval of the volume object by siibra after the user asks for the
   volume via `fetch()` method. When `fetch()` is invoked it accesses to
   corresponding volume provider based on the specifications given by volume
   index, fragment, z, label, variant, and format. According to the source type,
   the provider invokes the correct class and fetches the data accordingly.

**Defaults**

* Volume with several variants: the first variant listed in the configuration is
  fetched. The user is informed along with a list of possible variants.
* Volume with several fragments: All fragments are retrieved and combined to
  provide the whole volume. (This may cause some array length issues on the user
  end so the user should be informed. Potentially, this may be changed to fetch
  only the first fragment along with info and a list of options.)

**Implementation Notes**

* When adjusting to a new type of data or special cases, it is highly encouraged
  to use one of the existing parameters.
* Always inform a user when there are options available and the default is chosen.

IV. Multimodal data features
============================

Adding data to siibra-toolsuite
-------------------------------

0. Is the feature type class representation for the data?

    * Yes: go to step 1.
    * No: create feature type subclass and PR to siibra-python main.

1. Is the feature type already described by the schema (in siibra-python/config_schema)?

    * Yes: go to step 2.
    * No: create schema and PR to siibra-python main.

2. Create feature jsons and create a PR to siibra-configurations.
3. After merging the PR, create new tag on siibra-configurations.
4. Bump siibra-python version to match the new tag.

Anatomical Anchor
-----------------


