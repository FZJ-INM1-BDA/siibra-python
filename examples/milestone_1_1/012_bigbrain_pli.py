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
Ultra-high resolution imaging of human fiber architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
# Images are linked to reference spaces by image registration,
# and queried based on their bounding boxes.
# We retrieve blockface imaging features anchored to BigBrain space,
# and compile a tabular overview.
bigbrain = siibra.spaces.get("bigbrain")
features_bf = siibra.features.get(bigbrain, "blockface")
feature_table = siibra.features.tabulate(features_bf, ["name"])
feature_table

# %%
blockface = feature_table[
    feature_table.name.str.contains('Connectome')
].iloc[0].feature
plotting.plot_img(
    blockface.fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
)
