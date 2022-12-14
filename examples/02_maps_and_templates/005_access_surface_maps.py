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
jubrain = siibra.parcellations["julich"]

# %%
# We can tell volumetric from surface spaces using their `is_surface` attribute.
for space in jubrain.spaces:
    if space.is_surface: print(space)

# %%
# We then select the desired space from the available spaces that provide surfaces
space = jubrain.spaces['fsaverage6']

# The surface meshes are accessed using the `get_template` method through spaces.
template = space.get_template()
print(template)

# we can then fetch the mesh by
surfmap = template.fetch()

# %%
# The returned structure is a dictionary of three numpy arrays representing the vertices, faces, and labels respectively. 
# Each vertex defines a 3D surface point, while the faces are triplets of indices into the list of vertices, defining surface triangles.
# The labels provide the label index associated with each vertex.
print(surfmap.keys())

# %%
# For plotting meshes, most python libraries can be employed.
# We recommend again the plotting module of `nilearn <https://nilearn.github.io>`_. 
from nilearn import plotting
plotting.view_surf((surfmap['verts'], surfmap['faces']))

# %%
# Note from the output of the get_template that, for surfaces, the `get_template()` method accepts an additional parameter
# 'variant' to select between avalbable surfaces, in this case,'white matter', 'pial' and 'inflated' surface (left and right for each).
template2 = space.get_template(variant="inflated/right")
print(template2)
surfmap2 = template2.fetch()
plotting.view_surf((surfmap2['verts'], surfmap2['faces']))

# %%
# We can alternatively get a template from siibra module as a handy shortcut and fetch the mesh
surfmap3 = siibra.get_template(space_spec="fsaverage6", variant="pial/left").fetch()
plotting.view_surf((surfmap3['verts'], surfmap3['faces']))