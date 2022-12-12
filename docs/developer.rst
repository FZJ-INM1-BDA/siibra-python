Developer documentation
=======================

Frequently used design patterns
-------------------------------

``siibra`` makes frequent use of some typical design patterns.

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

