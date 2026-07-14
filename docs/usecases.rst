.. _usecases:

=========
Use cases
=========

`siibra-python` supports neuroscience workflows that need anatomical context,
atlas-based interpretation, multimodal data access, or reproducible links
between brain regions, coordinate spaces, maps, and measurements.

This page gives a research-oriented overview of possible use cases. For the
concepts behind these workflows, see :ref:`concepts`. For executable examples,
see :ref:`examples`.

Spatial interpretation of findings
==================================

* **Localizing experimental results:**
  Relate coordinates, image clusters, lesions, stimulation sites, electrode
  contacts, or segmentation results to named brain areas.

* **Comparing findings across studies:**
  Use common reference spaces and atlas definitions to make results from
  different studies easier to compare.

* **Handling anatomical uncertainty:**
  Represent assignments probabilistically instead of forcing spatial findings
  into a single deterministic label.

Relevant examples:

* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`
* :doc:`Assign modes in activation maps to brain regions <examples/05_anatomical_assignment/002_activation_maps>`
* :doc:`Anatomical characterization and multimodal profiling of regions of interest <examples/tutorials/2025-paper-fig3>`

Brain organization across scales
================================

* **Linking macrostructure to microstructure:**
  Connect MRI-scale findings with cytoarchitectonic maps, cortical layers, and
  high-resolution histological resources.

* **Studying regional specialization:**
  Compare brain areas by cellular, molecular, connectivity, functional, and
  anatomical properties.

* **Exploring multilevel brain organization:**
  Move between whole-brain organization, regional architecture, and local
  tissue-level measurements.

Relevant examples:

* :doc:`Accessing parcellation maps <examples/02_maps_and_templates/003_accessing_maps>`
* :doc:`Spatial properties of brain regions <examples/01_atlases_and_parcellations/005_brain_region_spatialprops>`
* :doc:`Cortical cell body distributions <examples/03_data_features/003_cell_distributions>`
* :doc:`Anatomically guided reproducible extraction of full resolution image data from cloud resources <examples/tutorials/2025-paper-fig5>`

Multimodal data integration
===========================

* **Combining heterogeneous measurements:**
  Bring together maps, images, tables, connectivity matrices, gene expression,
  receptor densities, and microscopy-derived data.

* **Building regional profiles:**
  Summarize multiple data modalities for a brain area or region of interest.

* **Finding anatomically relevant datasets:**
  Discover data resources linked to a brain area, coordinate, bounding box, or
  image-defined region.

Relevant examples:

* :doc:`Understanding links between data features and anatomical locations <examples/03_data_features/000_matchings>`
* :doc:`Neurotransmitter receptor densities <examples/03_data_features/001_receptor_densities>`
* :doc:`Gene expressions <examples/03_data_features/004_gene_expressions>`
* :doc:`Connectivity matrices <examples/03_data_features/006_connectivity_matrices>`
* :doc:`Compound features <examples/03_data_features/009_compound_features>`
* :doc:`Multimodal comparison of two cortical brain areas <examples/tutorials/2025-paper-fig4>`

Neuroimaging analysis support
=============================

* **Annotating imaging results:**
  Add anatomical labels, probabilities, overlaps, and atlas metadata to fMRI,
  MRI, DTI, lesion, or stimulation analyses.

* **Evaluating custom maps or segmentations:**
  Compare new parcellations, masks, or cluster maps with established anatomical
  references.

* **Creating atlas-informed derivatives:**
  Generate region-wise summaries, masks, lookup tables, or feature tables for
  downstream analysis.

Relevant examples:

* :doc:`Assigning coordinates to brain regions <examples/05_anatomical_assignment/001_coordinates>`
* :doc:`Assign modes in activation maps to brain regions <examples/05_anatomical_assignment/002_activation_maps>`
* :doc:`Anatomical characterization and multimodal profiling of regions of interest <examples/tutorials/2025-paper-fig3>`
* :doc:`Case study: Anatomical evaluation of subcortical maps <examples/tutorials/2025-paper-fig6>`

High-resolution and big-data workflows
======================================

* **Targeted access to large image resources:**
  Retrieve only the relevant part of a large microscopy or whole-brain image
  resource.

* **Microscopy-guided validation:**
  Inspect whether macroscopic findings correspond to cytoarchitectonic or
  histological structure at finer scale.

* **Sampling tissue regions reproducibly:**
  Extract anatomically defined image patches or volumes for cell detection,
  laminar analysis, or local tissue characterization.

Relevant examples:

* :doc:`Access BigBrain high-resolution data <examples/02_maps_and_templates/004_accessing_bigbrain>`
* :doc:`Access to BigBrain cortical layer meshes <examples/02_maps_and_templates/006_cortical_layer_meshes>`
* :doc:`Anatomically guided reproducible extraction of full resolution image data from cloud resources <examples/tutorials/2025-paper-fig5>`

Computational modeling and simulation
=====================================

* **Parameterizing models with anatomical data:**
  Use regional receptor, connectivity, cellular, or structural measurements as
  model parameters.

* **Adding regional heterogeneity to models:**
  Move beyond homogeneous brain models by assigning area-specific biological
  properties.

* **Interpreting model outputs anatomically:**
  Relate simulated activity, connectivity, or regional effects back to
  atlas-defined brain areas and data features.

Relevant examples:

* :doc:`Neurotransmitter receptor densities <examples/03_data_features/001_receptor_densities>`
* :doc:`Connectivity matrices <examples/03_data_features/006_connectivity_matrices>`
* :doc:`Parcellation-based functional data <examples/03_data_features/008_functional_timeseries>`
* :doc:`Multimodal comparison of two cortical brain areas <examples/tutorials/2025-paper-fig4>`

Machine learning and data science
=================================

* **Creating anatomically grounded feature vectors:**
  Combine multimodal measurements into region-wise tables for clustering,
  prediction, classification, or exploratory analysis.

* **Supporting interpretable models:**
  Link model features and outputs to anatomical areas, spatial locations, and
  biological measurements.

* **Preparing curated training or validation data:**
  Use atlas definitions and linked datasets to assemble reproducible,
  anatomy-aware datasets.

Relevant examples:

* :doc:`Comparative analysis of brain organisation in two brain regions <examples/03_data_features/007_comparative_assessment>`
* :doc:`Compound features <examples/03_data_features/009_compound_features>`
* :doc:`Multimodal comparison of two cortical brain areas <examples/tutorials/2025-paper-fig4>`
* :doc:`Anatomical characterization and multimodal profiling of regions of interest <examples/tutorials/2025-paper-fig3>`

Data curation and atlas extension
=================================

* **Integrating project-specific data:**
  Use local images, maps, masks, or measurements together with established atlas
  content.

* **Preparing data for shared atlas resources:**
  Prototype configuration files before contributing maps, templates, or data
  features to shared configurations.

* **Keeping data linked to anatomical context:**
  Preserve the relationship between a dataset, its spatial location, reference
  space, and anatomical interpretation.

Relevant examples and documentation:

* :doc:`Adding a custom parcellation map <examples/02_maps_and_templates/007_adding_custom_parcellation>`
* :doc:`Configuring atlas content <create_preconfiguration>`
* :doc:`Understanding links between data features and anatomical locations <examples/03_data_features/000_matchings>`

Education, exploration, and communication
=========================================

* **Teaching brain atlas concepts:**
  Demonstrate spaces, templates, maps, parcellations, brain areas, and
  multimodal features in executable notebooks.

* **Moving from visual exploration to code:**
  Use visual inspection in siibra-explorer and reproduce selected atlas queries
  in Python.

* **Communicating anatomical context:**
  Report findings with explicit atlas, space, map, region, and data provenance
  information.

Relevant examples:

* :doc:`Selecting preconfigured parcellations <examples/01_atlases_and_parcellations/001_accessing_parcellations>`
* :doc:`Find brain regions in a parcellation <examples/01_atlases_and_parcellations/003_find_regions>`
* :doc:`Basic brain region properties <examples/01_atlases_and_parcellations/004_brain_region_metadata>`
* :doc:`Utilizing locations of interest <examples/04_locations/000_employing_locations_of_interest>`
