.. _usecases:

=========
Use cases
=========

``siibra-python`` supports neuroscience workflows that need anatomical context,
atlas-based interpretation, multimodal data access, or reproducible links
between brain regions, coordinate spaces, maps, and measurements.

This page gives a research-oriented overview of possible use cases. For the
concepts behind these workflows, see :ref:`glossary`. For executable examples,
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

Relevant examples are collected under anatomical assignment:
:ref:`examples`.


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

See :doc:`create_preconfiguration` for information about local and private
configuration files.


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