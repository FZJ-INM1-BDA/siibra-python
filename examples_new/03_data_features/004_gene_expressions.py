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
import siibra
from nilearn import plotting
from siibra.locations import PointCloud

# %%
# We select a brain region and query for expression levels of GABARAPL2.
region = siibra.get_region("julich 2.9", "V1")
features = siibra.find_features(
    region, siibra.modality_types.GENE_EXPRESSIONS,
    genes=["GABARAPL2"]
)

feature, *_ = features[0]
# Take a peek at how the data looks
feature.data[0].head()

# %%
# Since gene expressions are spatial features,
# let's check the reference space of the results.
space = feature.locations[0].space
print(space.name)

# %%
# Plot the locations of the probes that were found, together with the region
# mask of V1.
loc = feature.locations[0]
assert isinstance(loc, PointCloud)

all_coords = [p for p in loc.coordinates]
masks = region.find_regional_maps(space)
display = plotting.plot_roi(masks[0].fetch())
display.add_markers(all_coords, marker_size=5)
