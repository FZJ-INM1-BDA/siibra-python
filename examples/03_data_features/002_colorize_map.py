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
receptor_features = siibra.get_features(jubrain, siibra.modalities.ReceptorDistribution)

# %%
# Colorizing a map requires a dictionary that maps region objects to numbers.
# We build a dictionary mapping Julich-Brain regions to average receptor
# densities measured for GABAA.
receptor = 'GABAA'
mapping = {
    jubrain.decode_region(f.regionspec) : f.fingerprint[receptor].mean
    for f in receptor_features if receptor in f.fingerprint.labels
}

# %%
# Now colorize the Julich-Brain maximum probability map and plot it.
colorized_map = jubrain.get_map(space='mni152').colorize(mapping)
from nilearn import plotting
plotting.plot_stat_map(
    colorized_map, cmap='viridis',
    title=f"Average densities available for {receptor}"
)

