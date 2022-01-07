# Copyright 2018-2021
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from siibra.core import parcellation
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# We select a brain region and query for expression levels of GABARAPL2.
region = atlas.get_region("V1", parcellation="julich brain 2.9 colin")
features = siibra.get_features(
    region, siibra.modalities.GENEEXPRESSION,
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
