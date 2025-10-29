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
Atlas of the marmoset brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
# An atlas of the marmoset brain has been integrated with siibra.
# One way to identify it is to inspect the species available in the table
# of parcellations.
parc_table = siibra.parcellations.dataframe
print("Species with available parcellations:")
print("\n".join(parc_table.species.unique()))
parc_table[parc_table.species.str.contains("Callithrix")]


# %%
# The marmoset atlas consists of a reference space and template,
# parcellation, and parcellation map.
marmoset_parc = siibra.parcellations.MARMOSET_NENCKI_MONASH_ATLAS_2020
marmoset_space = siibra.spaces.MARMOSET_NENCKI_MONASH_TEMPLATE_NISSL_HISTOLOGY_2020
marmoset_template = marmoset_space.get_template()
marmoset_map = marmoset_parc.get_map(marmoset_space)

# Each object has the usualy metadata and DOI links, such as the space:
print(marmoset_space.name)
print(marmoset_space.description)
print(marmoset_space.urls)
print(marmoset_space.LICENSE)


# %%
# The marmoset parcellation has a detailed region hierarchy which can be
# rendered with `render_tree` method.
marmoset_parc = siibra.parcellations["Marmoset Nencki-Monash Atlas (2020)"]
marmoset_parc.render_tree()

# %%
# Plot the template
template_img = marmoset_space.get_template().fetch(max_bytes=1024**3)
plotting.view_img(
    template_img,
    bg_img=None,
    cmap="gray",
    black_bg=False,
    colorbar=False,
)


# %%
# The leaf regions of the parcellation is mapped on the space and has its own
# colormap, which can be displayed with nilearn
template_img = marmoset_template.fetch(max_bytes=1024**3)
cmap = marmoset_map.get_colormap()
map_img = marmoset_map.fetch(max_bytes=1024**3)
plotting.view_img(
    map_img,
    bg_img=template_img,
    cmap=cmap,
    title=marmoset_map.name,
    black_bg=False,
    colorbar=False,
)

# %%
# Furthermore, morphometry volumes are avaiable as data features for the marmoset atlas.
morphometry_volumes = siibra.features.get(marmoset_space, "morphometry")
for v in morphometry_volumes:
    plotting.plot_stat_map(
        v.fetch(max_bytes=1024**3),
        bg_img=template_img,
        black_bg=False,
        title=v.name,
        threshold=0,
    )
