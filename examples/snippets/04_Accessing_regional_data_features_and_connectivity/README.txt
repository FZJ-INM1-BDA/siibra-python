.. _features:

Retrieving region specific multimodal datasets
----------------------------------------------

``siibra`` provides access to data features of different modalities using its ``get_features()`` method, which accepts an **anatomical concept** like a brain region, parcellation, or reference space, and a feature modality as listed in the ``siibra.modalities`` registry. Currently available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, high-resolution volumes of interest, gene expressions, and cell distributions. Additional features, including functional activation maps and electrophysiologal recordings, will become available soon. ``siibra`` implements a unified handling for querying different types of features, broadly categorized into

 	- *spatial features*,  which are linked to atlas regions via coordinates (like contact points of physiological electrodes); 
 	- *regional features*, which are linked to atlases via a brain region specifications, like cell densities or neurotransmitter distributions; and 
 	- *parcellation features*, which are linked to an atlas via a whole brain parcellation, like connectivity matrices. 



