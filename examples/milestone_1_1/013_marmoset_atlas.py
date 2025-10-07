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
Marmoset Atlases
~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
# List preconfigured parcellations in siibra to see the new atlases and
# their species
siibra.parcellations.dataframe

# %%
# Callithrix jacchus, marmoset, atlas is an enitrely new species added. The
# atlas consists of a reference space, a parcellation, and a volume mapping the
# leaves of the parcellation. We can display the associated description,
# license, and urls/dois and show the template image of the space as usual.
marmoset_space = siibra.spaces["marmoset"]
print(marmoset_space.name)
print(marmoset_space.description)
print(marmoset_space.urls)
print(marmoset_space.LICENSE)

# fetch the image
template_img = marmoset_space.get_template().fetch(max_bytes=1024**3)
plotting.view_img(
    template_img,
    bg_img=None,
    cmap="gray",
    black_bg=False,
    colorbar=False,
)

# %%
# The marmoset parcellation has a detailed region hierarchy which can be
# rendered with `render_tree` method.
marmoset_parc = siibra.parcellations["Marmoset Nencki-Monash Atlas (2020)"]
marmoset_parc.render_tree()

# %%
# The leaf regions of the parcellation is mapped on the space and has its own
# colormap, which can be displayed with nilearn
marmoset_map = marmoset_parc.get_map(marmoset_space)
cmap = marmoset_map.get_colormap()
img = marmoset_map.fetch(max_bytes=1024**3)
plotting.view_img(
    img,
    bg_img=template_img,
    cmap=cmap,
    title=marmoset_map.name,
    black_bg=False,
    colorbar=False,
)

# %%
# Moreover, the atlas includes morphometry volumes which can be queried as
# features and plotted the same way as other image features
morphometry_volumes = siibra.features.get(marmoset_space, "morphometry")
for v in morphometry_volumes:
    plotting.plot_stat_map(
        v.fetch(max_bytes=1024**3),
        bg_img=template_img,
        black_bg=False,
        title=v.name,
        threshold=0,
    )
