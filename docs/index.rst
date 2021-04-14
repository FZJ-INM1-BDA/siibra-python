.. siibra documentation master file, created by
   sphinx-quickstart on Thu Oct  1 13:30:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../images/siibra-python.jpeg
  :width: 450
  :class: center
  :alt: siibra-python logo

======================================================
Software interface for interacting with brain atlases 
======================================================

Copyright 2020-2021, Forschungszentrum Jülich GmbH

*Authors: Big Data Analytics Group, Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH*

.. warning::
   siibra-python is still at an experimental stage. The API of the library is not stable, and the software is not yet fully tested. You are welcome to install and test it, but be aware that you will likely encounter bugs


``siibra`` is a Python client for interacting with "multilevel" brain atlases, which combine multiple brain parcellations and neuroscience datasets across different reference template spaces. It is designed to allow safe and convenient interaction with brain regions from different parcellations, to provide streamlined access to multimodal data features linked to brain regions, and to perform basic analyses of region-specific data features. The intention of ``siibra``  is to unify interaction with brain atlas data at different spatial scales, including parcellations and datasets at the millimeter scale in MNI space, as well as microstructural maps and microscopic data in the BigBrain space.


This software code is funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No.  945539 (Human Brain Project SGA3).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   authentication
   usage
   acknowledgements


..
    Indices and tables
    ------------------
    
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

