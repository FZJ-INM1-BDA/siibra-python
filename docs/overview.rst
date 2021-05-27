
========
Overview
========

``siibra`` is a Python client for interacting with "multilevel" brain atlases, which combine multiple brain parcellations and neuroscience datasets across different reference template spaces. It is designed to allow safe and convenient interaction with brain regions from different parcellations, to provide streamlined access to multimodal data features linked to brain regions, and to perform basic analyses of region-specific data features. The intention of ``siibra``  is to unify interaction with brain atlas data at different spatial scales, including parcellations and datasets at the millimeter scale in MNI space, as well as microstructural maps and microscopic data in the BigBrain space.

``siibra`` is largely developed inside the `Human Brain Project <https://humanbrainproject.eu>`_ for accessing the `human brain atlas of EBRAINS <https://ebrains.eu/service/human-brain-atlas>`_. 
It retrieves most of its concept and datasets from the `EBRAINS Knowledge Graph <https://kg.ebrains.eu>`_, and is designed to support the recently established `OpenMINDS metadata standards <https://github.com/HumanBrainProject/openMINDS_SANDS>`_, but also gives access to additional maps and resources.

The functionality of ``siibra-python`` matches common actions known from browsing the `EBRAINS Interactive Atlas Viewer <https://atlases.ebrains.eu/viewer>`_: Selecting different parcellations, browsing and searching brain region hierarchies, downloading maps, selecting regions, and accessing multimodal data features associated with brain regions. 
A key feature is a streamlined implementation of performing structured data queries for selected brain regions, which gives access to multimodal regional “data features”. 
``siibra`` implements a hierarchy of features, which unifies handling of *spatial data features* (which are linked to atlas regions via coordinates; like contact points of physiological electrodes), *regional data features* (which are linked to atlases via a brain region specifications, like cell densities or neurotransmitter distributions) and *global data features* (linked to an atlas via the whole brain or parcellation, like connectivity matrices or activation maps). 
As a result, all forms of data features can be queried using the same mechanism (``get_features()``) which takes as an argument a specification of the desired data modality, and respects the current selections made in the atlas. 
Currently, available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, gene expressions, and spatial properties of brain regions.
Additional features, including distributions of neuronal cells, functional activation maps and electrophysiologal recordings, will become available soon.

``siibra`` hides much of the complexity that would be required to interact with the individual data repositories that host the associated data.
By encapsulating many aspects of interacting with different maps and reference templates spaces, it also minimizes common risks like misinterpretation of coordinates from different reference spaces, or utilisation of inconsistent versions of parcellation maps. 
It aims to provide a safe way of using maps defined across multiple spatial scales. 

