.. _features:

Multimodal data features
------------------------

``siibra`` provides access to data features of different modalities using ``siibra.features.get()``, which accepts an **anatomical concept** like a brain region, parcellation, or reference space, and a feature type from the ``siibra.features`` module.
You can see the available feature types using `print(siibra.features.TYPES)`. 
Currently available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, high-resolution volumes of interest, gene expressions, and cell distributions. 
Additional features, including functional activation maps and electrophysiological recordings, will become available soon. 

