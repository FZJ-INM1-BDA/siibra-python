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
sulci atlas
~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
sulci_atlas = siibra.parcellations.get("sulci atlas")
sulci_atlas.render_tree()

# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 9))
for i, space in enumerate(
    siibra.maps.dataframe.query(f'parcellation == "{sulci_atlas.name}"')["space"]
):
    mp = sulci_atlas.get_map(space)
    cmap = mp.get_colormap()
    img = mp.fetch(resolution_mm=-1)
    plotting.plot_img(
        img=img,
        bg_img=mp.space.get_template().fetch(resolution_mm=0.4, max_bytes=3 * 1024**3),
        cmap=cmap,
        title=mp.name,
        axes=axs[i],
    )
    mesh = mp.fetch(region="Right calloso-marginal posterior fissure", format="mesh")
    plotting.view_surf(
        surf_mesh=[mesh["verts"], mesh["faces"]],
    )
plt.show()
