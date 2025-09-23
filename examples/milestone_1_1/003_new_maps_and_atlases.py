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
New Maps and Atlases
~~~~~~~~~~~~~~~~~~~~
"""

# %%
# List all parcellations in siibra to see the new atlases
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

for p in siibra.parcellations:
    print(p)

# %%
# There are 3 new parcellations. We make a list from the parcellations
# and display associated description, license, and urls/dois
new_parcellations = [
    siibra.parcellations["Combined Macaque Brain Atlas: MEBRAINS, fMRI, CHARM"],
    siibra.parcellations["Marmoset Nencki-Monash Atlas (2020)"],
    siibra.parcellations["Sulci atlas"],
]
for parcellation in new_parcellations:
    print(parcellation.name)
    print(parcellation.description)
    print(parcellation.urls)
    print(parcellation.LICENSE)

# %%
# Some parcellations are mapped in several spaces. Therefore, loop through the
# map registry while filtering for these parcellations. Then, fetch the images
# for each and plot them on their respective space templates.

# define coordinates to view the maps
cut_coords = {
    "MEBRAINS population-based monkey template": (13, -5, 16),
    "Marmoset Nencki-Monash Template (Nissl Histology) 2020": (-4, 11, 10),
    "BigBrain microscopic template (histology)": (19, 37, 26),
    "MNI Colin 27": (9, 50, 34),
    "MNI 152 ICBM 2009c Nonlinear Asymmetric": (11, 52, 30),
}
# plot each in a subplot
fig, axs = plt.subplots(5, 1, figsize=(15, 35))
i = 0
for parcellation in new_parcellations:
    mapped_spaces = siibra.maps.dataframe.query(
        f'parcellation == "{parcellation.name}"'
    )["space"]
    for space in mapped_spaces:
        mp = parcellation.get_map(space)
        cmap = mp.get_colormap()
        fetch_kwargs = {"max_bytes": 0.4 * 1024 ** 3} if "neuroglancer/precomputed" in mp.formats else {}
        img = mp.fetch(**fetch_kwargs)
        template_img = siibra.get_template(space).fetch(resolution_mm=1)
        plotting.plot_roi(
            img,
            bg_img=template_img,
            cmap=cmap,
            title=f"{mp.name}\n{space}\nnumber of regions: {len(mp.regions)}",
            axes=axs[i],
            black_bg=False,
            cut_coords=cut_coords[space],
            colorbar=False,
        )
        i += 1
