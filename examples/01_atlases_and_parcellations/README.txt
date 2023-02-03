.. _atlases:

Atlases and brain parcellations
-------------------------------

Atlases are the most high-level concepts in ``siibra``. 
They do not provide much specific functionality, instead they mostly provide a common context for a collection of parcellations and reference spaces of the same species, which are integrated with each other through semantic and spatial links. Atlas objects also provide some shortcuts for using functionalities of their parcellations, parcellation maps and spaces.

.. _parcellations:

Similar to atlases, brain parcellations are semantic objects as well. The define a hierarchy of brain regions, and provide access to available parcellations maps in different reference spaces. 
Predefined atlases and parcellations can be accessed via `siibra.atlases` and `siibra.parcellations`, which provide an autocomplete functionality in the interactive interpreter, and allow to fetch via keyword matching using their `.get()` methods, e.g. you can run `human_atlas = siibra.atlases.get('human')` and `julich_brain = siibra.parcellations.get('julich')`.

