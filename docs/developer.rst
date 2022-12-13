Developer documentation
=======================


Object instantiation
--------------------

Since siibra creates many objects in a typical workflow, the creation of objects is following a clear structure.
There are two main approaches of object creation:

1. Preconfigured objects, via the configuration module. This happens automatically for
    - subclasses of `AtlasConcept`, if the class parameter `configuration_folder` points to a subfolder of siibra configuration repositories.
    - subclasses of `Feature`, if the class parameter `configuration_folder` points to a subfolder of siibra configuration repositories.
   the implementation of these queries is centralized in `AtlasConcept.get_instances()` and `Feature.get_instances()`, respectively,
   and relies on the registrations done in `AtlasConcept.__init_subclass__()` and `Feature.__init_subclass__()`.
2. Live queries, via the livequery module. This applies to subclasses of `Feature`, and relies on subclasses of `LiveQuery` 
   with the class parameter `FeatureType` set to the corresponding subclass of `Feature`. This triggers the query to be registered
   in the `Feature._live_queries` list via `LiveQuery.__init_subclass__()`. Any registered live queries will then automatically be called 
   in `Feature.get_instances()`, extending the list of possible preconfigured object instances.



Note that `AtlasConcept` keeps the list of instances in `InstanceTable`s during runtime, which can be accessed via `AtlasConcept.registry()` for any subclass.
Instances of `Feature` subclasses are just stored as a list at class level.


Frequently used design patterns
-------------------------------

``siibra`` makes frequent use of some typical design patterns.


**Importing local classes**

Often, siibra implementations make use of classes from different siibra modules.
It is common Python practice to do all imports at the start of a file, and not locally inside functions / classes.

Direct imports of classes from other local modules like here:

.. code-block:: python
    :caption: Imports of local classes (should be avoided)

    from ..core.space import Space

    def my_func():
        s = Space()


result in immediate creation of the class, and quickly leads to cyclic import dependencies which result in runtime errors.

This effect can often be avoided by only importing the siibra module, and deferring the class creating to a later stage in the code:

.. code-block:: python
    :caption: Imports of local modules

    from ..core import space

    def my_func():
        s = space.Space()


In general, it seems a good practice to import specific classes only in the `__init__.py` files, 
and use module imports in other python files.
However, this rule of thumb is not yet consistently implemented and verified in siibra. 


 **Lazy loading via object properties.**

Since siibra pre-configures many objects, of which the user will typically only use a few 
(e.g. after filternig data features by brain regions), it is important that time and/or memory 
consuming operations are only executed when objects are actually requested and used. 
We typically solve this by implementing object properties with a lazy loading mechanism,
following this scheme:

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

