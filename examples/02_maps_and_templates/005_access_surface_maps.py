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

``siibra`` also provides basic access to surfaces. A popular reference space
for the human brain is the freesurfer fsaverage surface. It comes in three
variants: white matter surface, pial surface, and inflated surface. Each is
shipped with left and right hemispheres separately.
"""

# %%
# For plotting meshes, most python libraries can be employed.
# We recommend the plotting module of `nilearn <https://nilearn.github.io>`_.
import siibra
from nilearn import plotting
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/02-005.png'

# %%
# Load the Julich-Brain parcellation.
julich_brain = siibra.parcellations.get("julich 2.9")

# %%
# We can tell volumetric from surface spaces using their `is_surface` attribute.
for space in julich_brain.spaces:
    if space.provides_mesh:
        print(space)

# %%
# The surface map is accessed using the `get map` method where we specify the
# space. Note that we call the method here on the parcellation object, while
# previous examples usually called it on an atlas object.
mp = julich_brain.get_map(space='fsaverage6')

# %%
# For surfaces, the `fetch()` method accepts an additional parameter 'variant'.
# If not specified, siibra displays the possible options as a list fetches the
# first one from the list. Now let us fetch a specific variant and also the
# hemisphere fragment
mesh = mp.fetch(variant="inflated", fragment="left")
mesh
# The returned structure is a Surface consiting of three numpy arrays representing the
# vertices, faces, and vertex_labels respectively. Each vertex defines a 3D surface
# point, while the faces are triplets of indices into the list of vertices,
# defining surface triangles. The labels provide the label index associated with
# each vertex.

# %%
# Most meshes are shipped with a color map which we can fetch from the map
# object by
julich_brain_cmap = mp.get_colormap()

# Now we can plot the mesh
plotting.view_surf(
    surf_mesh=[mesh['verts'], mesh['faces']],
    surf_map=mesh['labels'],
    cmap=julich_brain_cmap, symmetric_cmap=False, colorbar=False
)
# %%
