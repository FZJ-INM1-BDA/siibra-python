# Copyright 2018-2021
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
Access to Big Brain cortical layer meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``siibra`` provides access to cortical layer meshes obtained from Big Brain.
"""

# %%
import siibra
from nilearn import plotting # for plotting
# As an alternative to previous examples, we can get the map we want in one line by
mp = siibra.parcellations["layers"].get_map(space="big brain")

# %%
# We can fetch the layer by speficifying the name.
mesh = mp.fetch("layer 1", format="neuroglancer/precompmesh")
plotting.view_surf((mesh['verts'], mesh['faces'])) # TODO: implement a basic color map

# %%
# We can also choose the hemisphere and also calculate the thickness between two layers.
# First, fetch two layers:
mesh_r_1 = mp.fetch("layer 1", format="neuroglancer/precompmesh", hemisphere="right")
mesh_r_7 = mp.fetch("non-cortical", format="neuroglancer/precompmesh", hemisphere="right")

# then calculate the thickness using the map obejct created above
thickness = mp.find_layer_thickness(mesh_r_1, mesh_r_7)

# Let us display the histogram of the cortical layer thickness
import matplotlib.pyplot as plt
plt.hist(thickness, 200)

# %%
# We can now show the mesh using the thickness surf map
plotting.view_surf((mesh_r_1['verts'], mesh_r_1['faces']),
    surf_map=thickness, symmetric_cmap=False, cmap='winter')