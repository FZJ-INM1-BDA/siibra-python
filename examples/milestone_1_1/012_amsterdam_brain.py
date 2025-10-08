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
Amsterdam Brain
~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
mni152 = siibra.spaces["bigbrain"]
transmittance = siibra.features.get(mni152, "transmittance")
for f in transmittance:
    print(f.name)
    print(f.modality)
plotting.plot_img(transmittance[0].fetch(), cmap="magma", title=transmittance[0].name)

# %%
# t2_weighted_150um = [
#     f
#     for f in siibra.features.get(
#         mni152,
#         siibra.features.generic.Image,
#         modality="T2-weighted (T2w) image",
#     )
#     if "150 micrometer" in f.name
# ][0]
# t2_weighted_150um_img = t2_weighted_150um.fetch()
# plotting.view_img(
#     t2_weighted_150um_img,
#     bg_img=None,
#     cmap="gray",
#     symmetric_cmap=False,
#     black_bg=True,
#     colorbar=False,
# )

# %%
# plotting.view_img(
#     fa_img,
#     bg_img=t2_weighted_150um_img,
#     cmap="magma",
#     symmetric_cmap=False,
#     opacity=0.6,
#     colorbar=False,
# )


# %% compare histograms between regions and modalities
# julich_brain = siibra.parcellations["julich 3.1"]
# regions = [
#     julich_brain.get_region(spec)
#     for spec in ["4p right", "hoc1 right", "3b right"]
# ]
# fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
# for i, r in enumerate(regions):
#     pmap = r.get_regional_map(mni152, "statistical")
#     samples = pmap.draw_samples(10000)
#     values = fractional_anisotropy.evaluate_points(samples)
#     axs[i].hist(values[values > 0], bins=50, density=True)
#     axs[i].set_title(r.name)
