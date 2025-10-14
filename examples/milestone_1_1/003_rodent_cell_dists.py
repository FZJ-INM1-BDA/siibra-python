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
# In order to see all available cell distribution features for mice, query with
# the whole mouse parcellation and display dataset information:
allen_mice_parc = siibra.parcellations["Allen Mouse v3 2017"]
mice_cell_dists = siibra.features.get(allen_mice_parc, "cell distribution")
print("Found:", len(mice_cell_dists), "cell distributions")
datasets = {
    "".join(ds.name for ds in cd.datasets): f"{cd.description}\nDOI: {''.join(cd.urls)}"
    for cd in mice_cell_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)

# %%
# These cell distributions are coming from the same dataset. To see what differs
# among these features we print the anchored regions and subject information
for cd in mice_cell_dists:
    print(f"Region: {cd.anchor}, subject specification: {cd.subject}")

# %%
# There are several smaller olfactory areas in mice brain, however, we can get
# cell distributions anchored in them by just querying for the lower level
# structure "Olfactory areas:
olfactory_areas = allen_mice_parc.get_region("Olfactory areas")
olfactory_areas.render_tree()
olfactory_areas_dists = siibra.features.get(olfactory_areas, "cell distribution")

# %%
# To display them, we first need the combined mask of all the sub-structures.
# This can be achieved with `volume.merge` method.
olfactory_areas_mask = siibra.volumes.volume.merge(
    [r.get_regional_mask("allen") for r in olfactory_areas]
).fetch()

# %%
# We can use the fact that there are several subjects in this dataset by
# plotting the cell distribution per subject. Furthermore, the cells can be
# easily colored based on their original anchor.
allen_mouse_template = siibra.get_template("mouse").fetch()
for subject in {s.subject for s in olfactory_areas_dists}:
    celldists_for_subject = [cd for cd in olfactory_areas_dists if cd.subject == subject]
    display = plotting.plot_roi(
        olfactory_areas_mask,
        bg_img=allen_mouse_template,
        colorbar=False,
        draw_cross=False,
        title=cd.subject,
        cut_coords=celldists_for_subject[0].data.mean()
    )
    cmap = plt.get_cmap("tab10", len(celldists_for_subject))
    for i, cd in enumerate(celldists_for_subject):
        display.add_markers(cd.data, marker_color=cmap(i), marker_size=1)

# %%
# In exact same way, the cell distributions for rat brain can be queried.
waxholm_rat_parc = siibra.parcellations["Waxholm Sprague Dawley rat brain (v4)"]
rat_cell_dists = siibra.features.get(waxholm_rat_parc, "cell distribution")
print("Found:", len(rat_cell_dists), "cell distributions")
datasets = {
    "".join(ds.name for ds in cd.datasets): f"{cd.description}\nDOI: {''.join(cd.urls)}"
    for cd in rat_cell_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)

for cd in rat_cell_dists:
    print(f"Region: {cd.anchor}, subject specification: {cd.subject}")

# %%
# Similarly, we can display the parvalbumin positive neurons in olfactory bulb
# for rat per subject
olfactory_bulb = waxholm_rat_parc.get_region("olfactory bulb")
olfactory_bulb_mask = olfactory_bulb.get_regional_mask("waxholm").fetch()
rat_olfactory_bulb_dists = siibra.features.get(olfactory_bulb, "cell distribution")
wax_rat_template = siibra.get_template("Waxholm").fetch()
for subject in {s.subject for s in rat_olfactory_bulb_dists}:
    celldists_for_subject = [cd for cd in rat_olfactory_bulb_dists if cd.subject == subject]
    display = plotting.plot_roi(
        olfactory_bulb_mask,
        bg_img=wax_rat_template,
        colorbar=False,
        draw_cross=False,
        title=cd.subject,
        cut_coords=celldists_for_subject[0].data.mean()
    )
    cmap = plt.get_cmap("jet", len(celldists_for_subject))
    for i, cd in enumerate(celldists_for_subject):
        display.add_markers(cd.data, marker_color=cmap(i), marker_size=1)
