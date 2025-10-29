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
High-resolution postmortem MRI scans of a human brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# A postmortem brain ("Chenonceau brain") underwent multimodal MRI scanning
# with different modalities at very high resolutions.
# The data was registered to the MNI reference space.
mni152 = siibra.spaces.get("mni152")
features_img = siibra.features.get(mni152, siibra.features.generic.Image)
feature_table = siibra.features.tabulate(features_img, ['name', 'modality'])
feature_table

# %%
# As an example, the fractional anisotropy map is fetched and plotted.

fa_img = feature_fa.fetch()
plotting.plot_img(fa_img, cmap="magma", title=feature_fa.name)feature_fa = (
    feature_table[feature_table.modality.str.contains("anisotropy")]
    .iloc[0].feature
)

# %%
# The T2-weighted scan of the same brain provides a more detailed 
# and appropriate background image to study the FA map.
# We plot the 150um version.
# TODO T2 not well visible, maybe plot_statmap is better suited for this?
t2_weighted_150um = feature_table[
    feature_table.modality.str.contains('T2')
    & feature_table.name.str.contains('150')
].iloc[0].feature
t2_weighted_150um_img = t2_weighted_150um.fetch()
plotting.view_img(
    fa_img,
    bg_img=t2_weighted_150um_img,
    cmap="magma",
    symmetric_cmap=False,
    opacity=0.6,
    colorbar=False,
)

# %%
# siibra can use the FA map to draw samples from specific regions of
# interest. As an example, we compare GABA/BZ densities in 4p right, hoc1 right, and
# 3b right.
julich_brain = siibra.parcellations.get("julich 3.1")
regions = [
    julich_brain.get_region(spec)
    for spec in ["4p right", "hoc1 right", "3b right"]
]
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
for i, r in enumerate(regions):
    pmap = r.get_regional_map(mni152, "statistical")
    samples = pmap.draw_samples(10000)
    values = feature_fa.evaluate_points(samples)
    axs[i].hist(values[values > 0], bins=50, density=True)
    axs[i].set_title(r.name)
