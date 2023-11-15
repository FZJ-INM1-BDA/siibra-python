..  _mainconcepts:
Elements of an atlas in siibra
==============================

A typical brain atlas consists of 

    * a reference coordinate space with a brain reference template, typically a structural image of the brain,
    * a parcellation map, which labels coordinates in the reference space with an identifier of a brain region,
    * a taxonomy, which defines the names of brain regions used by the parcellation, and links them to the identifiers used in the parcellation map.

siibra extends this basic concept by linking multiple spaces and parcellation maps with complementary properties, and
combining them with multimodal data features that capture characteristic aspects of brain organisation for locations in the brain.

Labelled maps vs probability maps
---------------------------------
The above describes what siibra calls a "labelled map", where each location in the brain is uniquely labelled to
belong to exactly one brain region. However, siibra also supports "statistical maps", where each location in the brain
is mapped to multiple brain regions with different weights (or probabilities). Each coordinate is then not mapped to a
single label, but to a vector of weights, one per brain region. The weights for a specific brain region across all
coordinates in the brain then form the statistical (or probabilistic) map for this single region, and the statistical
maps of multiple regions may overlap.#f Datasets linked to locations in the brain


Multiple reference spaces
-------------------------
siibra supports different parcellation maps for a given reference space, but also different reference coordinate
spaces for a given species. The human brain atlas in EBRAINS provides parcellation maps in 

    * the MNI 152 space [#f1]_, which is defined by a multi-subject average of structural MRI scans defined at a
      resolution of about 1mm,
    * the freesurfer fsaverage space [#f2]_, which is a pure surface space defined over the vertices of a surface mesh of
      an average brain,
    * the BigBrain histological space [#f3]_ which is the anatomical space of a single subject brain that was 3D
      reconstructed from more than 7000 histological sections at an isotropic resolution of 20 micrometers.

siibra-explorer is designed visualize any of these different types efficiently, by allowing to zoom into very high
resolution images, and by offering both volumetric and surface-based viewing modes.

Relationships between spaces
----------------------------
Some parcellations maps, especially the Julich-Brain cytoarchitectonic maps [#f4]_, are available in all those spaces,
this way providing a natural link across those spaces and different spatial scales. An additional link is available
through spatial transformations [#f5]_, which map coordinates in one space to their most likely corresponding coordinate
in another. These spatial transformations are used by siibra-explorer when you change the reference space, in order to
preserve the currently viewed region of interest.

Datasets linked to locations in the brain
-----------------------------------------
siibra provides access to data features anchored to locations in the brain. Locations can be defined in very different
ways, such as  by specification of a brain region (thus only providing a semantic definition), a coordinate in a
reference space, or a bounding box in a reference space. Data features represent datasets hosted on a public repository,
typically but not exclusively the EBRAINS Knowledge Graph. A growing share of linked datasets are directly interpreted
by siibra-explorer, which means that siibra-explorer offers direct access to the underlying data: Further than just
displaying information about the dataset, siibra can visualize the data itself and allows to download it. These directly
interpreted features are categorized into molecular, cellular, functional, fibres, connectivity and macrostructural.
Many additional datasets are linked to brain regions, which only provide a metadata description and link to the
corresponding dataset page on their original repository. 


.. rubric::
    :fontSize: 7

.. [#f1] Fonov V, Evans A, McKinstry R, Almli C, Collins D. Unbiased nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage. 2009;47:S102. doi:10.1016/S1053-8119(09)70884-5. *More precisely, siibra supports the MNI ICBM 152 2009c nonlinear asymmetric template, as well as the Colin 27 single subject average.*
.. [#f2] Dale AM, Fischl B, Sereno MI. Cortical Surface-Based Analysis: I. Segmentation and Surface Reconstruction. NeuroImage. 1999;9(2):179-194. doi:[10.1006/nimg.1998.0395](https://doi.org/10.1006/nimg.1998.0395)
.. [#f3] Amunts K, Lepage C, Borgeat L, Mohlberg H, Dickscheid T, Rousseau ME, Bludau S, Bazin PL, Lewis LB, Oros-Peusquens AM, Shah NJ, Lippert T, Zilles K, Evans AC. BigBrain: An Ultrahigh-Resolution 3D Human Brain Model. Science. 2013;340(6139):1472-1475. doi:[10.1126/science.1235381](https://doi.org/10.1126/science.1235381)#f Relationships between different reference spaces
.. [#f4] Amunts K, Mohlberg H, Bludau S, Zilles K. Julich-Brain: A 3D probabilistic atlas of the human brain's cytoarchitecture. Science. 2020;369(6506):988-992. doi:[10.1126/science.abb4588](https://doi.org/10.1126/science.abb4588)
.. [#f5] Lebenberg J, Labit M, Auzias G, Mohlberg H, Fischer C, Rivière D, Duchesnay E, Kabdebon C, Leroy F, Labra N, Poupon F, Dickscheid T, Hertz-Pannier L, Poupon C, Dehaene-Lambertz G, Hüppi P, Amunts K, Dubois J, Mangin JF. A framework based on sulcal constraints to align preterm, infant and adult human brain images acquired in vivo and post mortem. Brain Struct Funct. 2018;223(9):4153-4168. doi:[10.1007/s00429-018-1735-9](10.1007/s00429-018-1735-9)

