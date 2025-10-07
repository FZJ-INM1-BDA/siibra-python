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
Rat receptor autoradiography
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
import matplotlib.pyplot as plt
from nilearn import plotting

waxhlom_space = siibra.spaces["waxholm"]

# %%
# Receptor autoradiography sections are anchored in to bounding boxes in
# reference spaces. To see all available options for rat, query for with waxholm
# space. Then, display dataset information:
autoradiography_sections = siibra.features.get(waxhlom_space, "autoradiography section")
datasets = {
    "".join(ds.name for ds in tcd.datasets): "".join(tcd.urls)
    for tcd in autoradiography_sections
}
for name, doi in datasets.items():
    print("Name:", name)
    print("DOI:", doi)

# %%
# The names of the datasets reveal that they are divided by neurotransmitters.
# For a detailed look at the serotonin, we query for sections intersecting with
# the region dorsal thalamus. `siibra` will automatically merge the masks of
# the children of dorsal thalamus create a mask to compare against the sections.
thalamus = siibra.get_region("waxholm", "dorsal thalamus")
sections = siibra.features.get(thalamus, "autoradiography section")
print("Sections found:", len(sections))
serotonin_sections_thalamus = [s for s in sections if "serotonin" in s.name]
print("Sections from the serotonin study:", len(serotonin_sections_thalamus))

# %%
# Next, display them using matplotlib:
fig = plt.figure(figsize=(10, 28))
for i, section in enumerate(serotonin_sections_thalamus):
    img = section.fetch()
    plt.subplot(13, 3, i + 1)
    plt.imshow(img.dataobj.squeeze().T, aspect="equal")
    plt.axis("off")


# %%
autoradiography_volumes = siibra.features.get(waxhlom_space, "autoradiography volume")
datasets = {
    "".join(ds.name for ds in tcd.datasets): "".join(tcd.urls)
    for tcd in autoradiography_volumes
}
for name, doi in datasets.items():
    print("Name:", name)
    print("DOI:", doi)


# %%
plotting.view_img(
    autoradiography_volumes[0].fetch(),
    bg_img=waxhlom_space.get_template().fetch(),
    cmap="magma",
    symmetric_cmap=False,
    title=autoradiography_volumes[0].name
)
