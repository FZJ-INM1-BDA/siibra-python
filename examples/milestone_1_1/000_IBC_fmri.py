# Copyright 2018-2025
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
IBC - fMRI fingerprints with tasks and contrast labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/milestone_1_1_functional_fingerprint.png'

# %%
# The functional fingerprints are tabular data features,
# providing fMRI measurements of brain areas under a range of cognitive tasks.
# For this example, we specificy a cytoarchitectonic brain region and
# find the matching fingerprint with a fast custom filter.
# The default `siibra.features.get()` approach will also work,
# but it is slower as it cannot assume an explicit match.
julichbrain = siibra.parcellations["julich 3.1"]
area3b_left = julichbrain.get_region("area 3b left")
functional_fingerprints = siibra.features.get(
    area3b_left,
    siibra.features.functional.FunctionalFingerprint,
    exact_match_only=True
)
# There is exactly one functional fingerprint given a region and parcellation
assert len(functional_fingerprints) == 1
area3b_left_fp = functional_fingerprints[0]

# %%
# The actual data is exposed as a pandas DataFrame with columns for the task and
# task label, as well as the signal strength.
# TODO Expose the unit of the values properly.
area3b_left_fp.data

# %%
# We plot the functional fingerprint as a horizontal bar chart,
# color-grouped by task.
area3b_left_fp.plot(backend="plotly")

# %%
# Finally, we select a specific task label, retrieve its values
# over all areas of the Julich-Brain atlas, and colorize a brain map in MNI space.
functional_fingerprints = siibra.features.get(
    julichbrain,
    siibra.features.functional.FunctionalFingerprint,
    exact_match_only=True
)
assert len(functional_fingerprints) == 1
julichbrain_functional_fingerprint = functional_fingerprints[0]
q = "task == 'ArchiSocial' & labels == 'triangle_mental-random'"
values_per_region = julichbrain_functional_fingerprint.data.query(q).iloc[0].to_dict()
colored_map = julichbrain.get_map("MNI 152").colorize(values_per_region)
plotting.view_img(
    colored_map.fetch(),
    cmap="magma",
    resampling_interpolation="nearest",
)
