"""
Gene expressions
~~~~~~~~~~~~~~~~

``siibra`` can query gene expression data from the Allen brain atlas. The gene
expressions are linked to atlas regions by coordinates of their probes in MNI
space. When querying feature by a region,  ``siibra`` automatically builds a
region mask to filter the probes. 

.. hint::
    This feature is used by the `JuGEx toolbox
    <https://github.com/FZJ-INM1-BDA/siibra-jugex>`_, which provides an
    implementation for differential gene expression analysis between two
    different brain regions as proposed by Bludau et al.
"""

# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# We select a brain region and query for expression levels of GABARAPL2.
region = atlas.get_region("V1")
features = siibra.get_features(
    region, siibra.modalities.GeneExpression,
    gene=siibra.features.gene_names.GABARAPL2)
print(features[0])

# %%
# Since gene expressions are spatial features,
# let's check the reference space of the results.
space = features[0].space
assert(all(f.space==space for f in features))

# %%
# Plot the locations of the probes that were found, together with the region
# mask of V1.
from nilearn import plotting
all_coords = [tuple(g.location) for g in features]
mask = region.build_mask(space)
display = plotting.plot_roi(mask)
display.add_markers(all_coords,marker_size=5)
