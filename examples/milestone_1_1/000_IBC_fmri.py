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
Functional fingerprints from IBC fMRI data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/ibc_thumbnail.png'

# %%
# Functional fingerprints are tabular data features linked to brain areas.
# They provide z-scores of functional activations in an area, measured by fMRI across a range of cognitive tasks.
# Here we query for functional fingerprints for Area 3b in the left hemisphere.
# We speed up the query by checking only for exact matches - this will avoid testing for approximately related data.
julichbrain = siibra.parcellations.get("julich 3.1")
area3b_left = julichbrain.get_region("area 3b left")
fingerprints = siibra.features.get(
    area3b_left,
    siibra.features.functional.FunctionalFingerprint,
    exact_match_only=True,
)

# We expect one functional fingerprint per brain area
assert len(fingerprints) > 0, "No functional fingerprint found for area 3b left"
fp = fingerprints[0]

# %%
# The actual data is exposed as a pandas DataFrame.
# The z-scores of functional activations in the brain area are indexed is a multi-index consisting of the task name and label.
# TODO The value column should include a unit specification, e.g. 'Activation for Area 3b left [z-score]'.
fp.data

# %%
# The default plot of a functional fingerprint is a bar chart, color-coded by task.
fp.plot(backend="plotly")

# %%
# Often, it is interesting to see a brain map of activation z-scores for a specific task label
# across all brain regions. This is achieved by retrieving fingerprints for the
# complete parcellation, and using the corresponding values to colorize the parcellation map
# in MNI space.
fingerprints = siibra.features.get(
    julichbrain,
    siibra.features.functional.FunctionalFingerprint,
    exact_match_only=True,
)
assert len(fingerprints) > 0, "No functional fingerprints found for Julich-Brain"
fp = fingerprints[0]

# collect values of specific task label across all regions
task = "ArchiSocial"
label = "triangle_mental-random"
values_per_region = fp.data.loc[(task, label)]

# colorize and plot the parcellation map
colored_map = julichbrain.get_map(space="MNI 152").colorize(values_per_region.to_dict())
view = plotting.plot_stat_map(
    colored_map.fetch(),
    cmap="magma",
    resampling_interpolation="nearest",
    cut_coords=area3b_left.compute_centroids("mni152")[0].coordinate,
)
view.title(f"{task}/{label} activations\n(average z-scores for {julichbrain.shortname} atlas)", size=10)
