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
Receptor autoradiography in rat brains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
import matplotlib.pyplot as plt
from nilearn import plotting

# %%
# Receptor autoradiography sections are anchored to reference spaces
# through image registration. A query thus uses the reference space as the
# starting point. Standard queries use the bounding box of each section, not
# the image content.
waxholm_space = siibra.spaces["waxholm"]
features_ar = siibra.features.get(waxholm_space, "autoradiography section")
feature_table = siibra.features.tabulate(
    features_ar,
    ["name", "datasets"],
    converters={"datasets": lambda ds: next(iter(ds))},
)
feature_table


# %%
# The features stem from a range of different datasets.
for dataset in feature_table.datasets.unique():
    print(dataset.name)
    print(dataset.urls[0]["url"])
    print()

# %%
# We can query for sections that intersect with a brain region. Here, sections
# intersecting with the rat dorsal thalamus are retrieved and filtered to
# serotonin autoradiography sections. `siibra` will automatically merge the
# masks of the children of dorsal thalamus create a mask to compare against
# the sections.
thalamus = siibra.get_region("waxholm", "dorsal thalamus")
sections = siibra.features.get(thalamus, "autoradiography section")
print("Sections found:", len(sections))
serotonin_sections_thalamus = [s for s in sections if "serotonin" in s.name]
print("Sections from the serotonin study:", len(serotonin_sections_thalamus))

# %%
# Display intersecting sections using matplotlib
fig = plt.figure(figsize=(7, 20))
for i, section in enumerate(serotonin_sections_thalamus):
    img = section.fetch()
    plt.subplot(13, 3, i + 1)
    plt.imshow(img.dataobj.squeeze().T, aspect="equal", cmap="jet")
    plt.axis("off")


# %%
# Similarly, we query for autoradiography volumes with the waxholm space and
# obtain the metadata.
autoradiography_volumes = siibra.features.get(waxholm_space, "autoradiography volume")
datasets = {
    "".join(
        f"Name: {ds.name}\nDescription: {ds.description}" for ds in tcd.datasets
    ): "".join(tcd.urls)
    for tcd in autoradiography_volumes
}
for desc, doi in datasets.items():
    print(desc)
    print("DOI:", doi)


# %%
# View the M2 receptor autoradiography volumes using nilearn.
M2_volume = autoradiography_volumes[0]
plotting.view_img(
    M2_volume.fetch(),
    bg_img=waxholm_space.get_template().fetch(),
    cmap="jet",
    symmetric_cmap=False,
    opacity=0.8,
)

# %%
# `siibra` can compute the intersection of volumes if they are in the same
# space. Therefore, we can obtain the mask of thalamus and find the intersection
# with the M2 autoradiography to extract the values corresponding only to
# thalamus, which can be visualized with matplotlib.
thalamus_mask = thalamus.get_regional_mask(waxholm_space)
intersection_mask_nii = M2_volume.intersection(thalamus_mask).fetch()
M2_vals_thalamus = M2_volume.fetch().get_fdata()[intersection_mask_nii.get_fdata() > 0]
plt.hist(M2_vals_thalamus.flatten(), bins=100)
