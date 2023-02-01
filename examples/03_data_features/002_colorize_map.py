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
Colorizing a map with neurotransmitter receptor densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parcellation maps can be colorized with a dictionary of values.
This is useful to visualize data features in 3D space.
Here we demonstrate it for average densities of the GABAA receptor.
"""

# %%
# We start by selecting an atlas and the Julich-Brain parcellation.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
jubrain = atlas.get_parcellation('julich 2.9')

# %%
# As in the previous example, we extract all receptor density features linked
# to Julich-Brain.
fingerprints = siibra.features.get(jubrain, siibra.features.molecular.ReceptorDensityFingerprint)
fingerprints[0].data

# %%
# Colorizing a map requires a dictionary that maps region objects to scalar values.
# We build a dictionary mapping Julich-Brain regions to average receptor
# densities measured for GABAA.
# Note: For each receptor fingerprint, from the regions it is anchored
# to, we consider those that belong to the julich brain atlas, and store
# the mean receptor density for each of their leaf nodes.
receptor = 'GABAA'
mapping = {
    c: fp.data['mean'][receptor]
    for fp in fingerprints if receptor in fp.receptors
    for r in fp.anchor.regions
    if r in jubrain
    for c in r.leaves
}

# %%
# Now colorize the Julich-Brain maximum probability map and plot it.
colorized_map = jubrain.get_map(space='mni152').colorize(mapping)
from nilearn import plotting
plotting.view_img(
    colorized_map, cmap='magma',
    title=f"Average densities available for {receptor}", symmetric_cmap=False
)

# %%
# Alternatively, we can display this map on a surface mesh using nilearn.
# Note that, you can switch between the hemispheres or variants (inflated or pial) from the plot itself.
plotting.view_img_on_surf(colorized_map, cmap='magma', symmetric_cmap=False, surf_mesh="fsaverage6")


# %%
