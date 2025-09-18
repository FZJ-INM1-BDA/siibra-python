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
Macaque neurotransmitter receptor density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_number = -1

# %%
mebrains = siibra.parcellations["MEBRAINS monkey"]
print(mebrains.name)

# %%
for f in siibra.features.get(mebrains, siibra.features.generic.Tabular):
    print("Modality:", f.modality)

# %%
receptor_density = siibra.features.get(
    mebrains,
    siibra.features.generic.Tabular,
    modality="Neurotransmitter receptor density",
)[0]
print(receptor_density.modality)

# %%
receptor_density.data

# %%
receptor = "kainate"

# %%
receptor_density.plot(
    kind="barh",
    y=receptor,
    ylabel="area",
    xlabel=f"{receptor} - {receptor_density.unit}",
    figsize=(10, 20),
)

# %%
mp = mebrains.get_map("mebrains")
mapped_regions = {}
for r in receptor_density.data.index:
    try:
        mapped_regions[r] = mebrains.get_region(r.replace("area_", "")).name
    except ValueError:
        print(f"Mask of {r} is not yet published.")

# %%
receptor_values = {}
for r, v in receptor_density.data[receptor].items():
    if r in mapped_regions:
        receptor_values[mapped_regions[r] + " left"] = v
        receptor_values[mapped_regions[r] + " right"] = v
receptor_density_map = mp.colorize(receptor_values, resolution_mm=-1)
plotting.view_img(
    receptor_density_map.fetch(),
    bg_img=mp.space.get_template().fetch(),
    cmap="magma",
    symmetric_cmap=False,
    resampling_interpolation="nearest",
)

# %%
for i, receptor in enumerate(receptor_density.data.columns):
    # plt.subplot((i + 1, 1))

    receptor_values = {}
    for r, v in receptor_density.data[receptor].items():
        if r in mapped_regions:
            receptor_values[mapped_regions[r] + " left"] = v
            receptor_values[mapped_regions[r] + " right"] = v
    receptor_density_map = mp.colorize(receptor_values, resolution_mm=-1)
    plotting.plot_stat_map(
        receptor_density_map.fetch(),
        bg_img=mp.space.get_template().fetch(),
        cmap="magma",
        resampling_interpolation="nearest",
    )
