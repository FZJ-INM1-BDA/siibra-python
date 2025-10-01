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

for p in siibra.parcellations:
    print(p)

# %%
# There are 3 new parcellations. We make a list from the parcellations
# and display associated description, license, and urls/dois
marmoset_parc = siibra.parcellations["Marmoset Nencki-Monash Atlas (2020)"]
print(marmoset_parc.name)
print(marmoset_parc.description)
print(marmoset_parc.urls)
print(marmoset_parc.LICENSE)

# %%
# Some parcellations are mapped in several spaces. Therefore, loop through the
# map registry while filtering for these parcellations. Then, fetch the images
# for each and plot them on their respective space templates.
marmoset_space = siibra.spaces["marmoset"]
mp = marmoset_parc.get_map(marmoset_space)
cmap = mp.get_colormap()
img = mp.fetch(max_bytes=1024**3)
template_img = marmoset_space.get_template().fetch(max_bytes=1024**3)
plotting.plot_roi(
    img,
    bg_img=template_img,
    cmap=cmap,
    title=mp.name,
    black_bg=False,
    colorbar=False,
)

# %%
morphometry_volumes = siibra.features.get(marmoset_space, "morphometry")
for vol in morphometry_volumes:
    plotting.plot_img(
        vol.fetch(max_bytes=1024**3),
        bg_img=template_img,
        black_bg=False,
        title=vol.name
    )
