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
Cell distributions in rodent brains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# In order to see all availabe cell distribution features for mice, query with
# the whole mouse parcellation and display dataset information
allen_mice_parc = siibra.parcellations[
    "Allen Mouse Common Coordinate Framework v3 2017"
]
all_mice_cell_dists = siibra.features.get(allen_mice_parc, "cell distribution")
print("Found:", len(all_mice_cell_dists), "cell distributions")
datasets = {
    "".join(ds.name for ds in cd.datasets): f"{cd.description}\nDOI: {''.join(cd.urls)}"
    for cd in all_mice_cell_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)

# %%
# At the moment, all mice cell distributions come from the same dataset. To see
# what differs among these features we print the anchored regions and subject
# information
for cd in all_mice_cell_dists:
    print(f"Region: {cd.anchor}, subject specification: {cd.subject}")

# %%
# In exact same way, the cell distributions for rat brain can be queried.
# Note that, at the moment one dataset investigating cell distributions is
# available, and the features differ in region and subjects:
waxholm_rat_parc = siibra.parcellations[
    "Waxholm Space atlas of the Sprague Dawley rat brain (v4)"
]
all_rat_cell_dists = siibra.features.get(waxholm_rat_parc, "cell distribution")
print("Found:", len(all_rat_cell_dists), "cell distributions")
datasets = {
    "".join(ds.name for ds in cd.datasets): f"{cd.description}\nDOI: {''.join(cd.urls)}"
    for cd in all_rat_cell_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)

for cd in all_rat_cell_dists:
    print(f"Region: {cd.anchor}, subject specification: {cd.subject}")
# %%
# Given the comprehensive nature of the datasets, one can query similar regions
# in rodents in order to observe potential similarities and differences in how
# the cells are distributed between species. We get the cell distributions
# for "olfactory bulb":
mice_olfactory_bulb_dists = [
    cd
    for cd in siibra.features.get(
        allen_mice_parc.get_region("main olfactory bulb"), "cell distribution"
    )
    if "-" not in cd.subject  # filter out specialized subselections
]
allen_mouse_template = siibra.get_template("mouse").fetch()
fig, axs = plt.subplots(len(mice_olfactory_bulb_dists), 1, figsize=(15, 24))
for i, cd in enumerate(mice_olfactory_bulb_dists):
    display = plotting.plot_img(
        img=allen_mouse_template,
        bg_img=None,
        cmap="gray",
        title=cd.name,
        cut_coords=cd.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
    )
    display.add_markers(cd.data, marker_color="r", marker_size=1)

# %%
# Similarly, display the cells in olfactory bulb for rat
rat_olfactory_bulb_dists = [
    cd
    for cd in siibra.features.get(
        waxholm_rat_parc.get_region("olfactory bulb"), "cell distribution"
    )
    if "-" not in cd.subject  # filter out specialized subselections
]
wax_rat_template = siibra.get_template("Waxholm").fetch()
fig, axs = plt.subplots(len(rat_olfactory_bulb_dists), 1, figsize=(15, 24))
for i, cd in enumerate(rat_olfactory_bulb_dists):
    display = plotting.plot_img(
        img=wax_rat_template,
        bg_img=None,
        cmap="gray",
        title=cd.name,
        cut_coords=cd.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
    )
    display.add_markers(cd.data, marker_color="b", marker_size=1)
