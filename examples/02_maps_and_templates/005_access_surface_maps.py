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
Access parcellation maps in surface space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``siibra`` also provides basic access to surfaces. 
A popular reference space for the human brain is the freesurfer fsaverage surface.
It comes in three variants: white matter surface, pial surface, and inflated surface.
"""

# %%
# Load the Julich-Brain parcellation.
import siibra
jubrain = siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9

# %%
# We can tell volumetric from surface spaces using their `is_surface` attribute.
for space in jubrain.spaces:
    if space.is_surface:
        print(space)
 
# %%
# The surface map is accessed in just the same way as volumetric maps, using the `get map` method.
# Note that we call the method here on the parcellation object, while previous examples usually
# called it on an atlas object.
# For surfaces however, the `fetch()` method accepts an additional parameter 'variant' to select
# between 'white matter', 'pial' and 'inflated' surface. 
surfmap = jubrain.get_map('fsaverage6').fetch(variant="inflated")

# %%
# The returned structure is a dictionary of three numpy arrays representing the vertices, faces, and labels respectively. 
# Each vertex defines a 3D surface point, while the faces are triplets of indices into the list of vertices, defining surface triangles.
# The labels provide the label index associated with each vertex.
print(surfmap.keys())

# %%
# For plotting meshes, most python libraries can be employes.
# We recommend again the plotting module of `nilearn <https://nilearn.github.io>`_. 
# We use Julich-Brain's native colormap for plotting.
from nilearn import plotting
jubrain_cmap = jubrain.get_colormap()
plotting.view_surf(
    surf_mesh = [surfmap['verts'], surfmap['faces']], 
    surf_map = surfmap['labels'], 
    cmap = jubrain_cmap, symmetric_cmap=False, colorbar=False
)