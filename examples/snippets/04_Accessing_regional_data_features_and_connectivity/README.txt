.. _features:

Retrieving region specific multimodal datasets
----------------------------------------------

``siibra`` implements unified handling for different types of features, namely

 - *spatial features* (which are linked to atlas regions via coordinates; like contact points of physiological electrodes), 
 - *regional features* (which are linked to atlases via a brain region specifications, like cell densities or neurotransmitter distributions), and 
 - *parcellation features* (linked to an atlas via a whole brain parcellation, like grouped connectivity matrices). 

As a result, all forms of data features can be queried using the same mechanism (``siibra.get_features()``) which accepts the specification of an concept (e.g. a selected brain region), and a data modality.
Currently available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, high-resolution volumes of interest, gene expressions, and cell distributions. 
Additional features, including functional activation maps and electrophysiologal recordings, will become available soon. Stay tuned!

TODO add examples


