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
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/02-006.png'
import siibra
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt

# %%
# Request the BigBrain cortical layer parcellation by Wagstyl et al.
# from siibra.
layermap = siibra.get_map(parcellation='layers', space="big brain")
layermap.regions


# %%
# We fetch the surface mesh of a layer by specifying a unique part of its name.
l4_surf_l = layermap.fetch(region="layer 4 left", format="mesh")

# %%
# We can also choose individual hemispheres, and recombine meshes.
# For illustration, we create a combined mesh of the white matter surface
# in the left hemisphere and the layer 1 surface of the right hemisphere.
wm_surf_r = layermap.fetch(region="non-cortical right", format="mesh")
mesh = siibra.commons.merge_meshes([l4_surf_l, wm_surf_r], labels=[10, 20])

# %%
plotting.view_surf(
    (mesh['verts'], mesh['faces']),
    surf_map=mesh['labels'],
    cmap='Set1', vmin=10, vmax=30, symmetric_cmap=False, colorbar=False
)

# %%
# Since the cortical layer surfaces share corresponding vertices,
# we can compute approximate layer thicknesses as Euclidean distances
# of mesh vertices. Here we compare the layer depth distribution of
# layers 4 and 5 across the left hemisphere.
plt.figure()
layers = [2, 4, 5]
thicknesses = {}
for layer in layers:
    upper_surf = layermap.fetch(f"layer {layer+1} left", format="mesh")
    lower_surf = layermap.fetch(f"layer {layer} left", format="mesh")
    thicknesses[layer] = np.linalg.norm(upper_surf['verts'] - lower_surf['verts'], axis=1)
    plt.hist(thicknesses[layer], 200)
plt.xlabel('Layer depth (left hemisphere)')
plt.ylabel('# vertices')
plt.legend([f"Layer {_}" for _ in layers])
plt.grid(True)

# %%
# We can plot the (last) thickness distribution on the surface
plotting.view_surf(
    (lower_surf['verts'], lower_surf['faces']),
    surf_map=thicknesses[4],
    symmetric_cmap=False, cmap='magma', vmax=0.3
)
# %%
