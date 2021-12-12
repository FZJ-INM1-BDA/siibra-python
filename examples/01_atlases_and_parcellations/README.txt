.. _atlases:

Atlases and brain parcellations
-------------------------------

Atlases are the most high-level concepts in ``siibra``. 
They provide a common context for a  collection of parcellations and reference spaces of the same species, which are integrated with each other through semantic and spatial links. Atlas objects also provide many convenient shortcuts for using functionalities of their parcellations, parcellation maps and spaces.

.. _parcellations:

Similar to atlases, brain parcellations are semantic objects. The define a hierarchy of brain regions, information about available parcellations maps in different reference spaces, and additional metadata about the parcellation. 

Predefined atlases and parcellations are accessed via the ``Registry`` objects ``siibra.atlases`` and ``siibra.parcellations``.

