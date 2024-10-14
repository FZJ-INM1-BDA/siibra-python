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
.. _templates:

Accessing brain reference templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other than a reference space, which is only a semantic entity, a reference template
is a spatial object representing the brain volume in the form of a 3D image or
3D surface mesh.
Since each reference template is uniquely linked to one particular brain reference
space, we access templates by the `get_template()` method of `Space` objects.

In `siibra`, reference templates are `Volume` objects.
Volumes allow fetching volumetric data from different types of "volume providers"
through their `fetch()` method.

It is important to note that the same volume (such as the BigBrain 3D model)
can be provided through different resources and formats,
which represent it as an image or surface mesh, each possibly through multiple
formats.
Therefore, a Volume object can have multiple volume providers, which are
selected depending on the parameters passed to `fetch()`.
This example will show some typical settings.
"""


# %%
import siibra
from nilearn import plotting
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/02-002.png'

# %%
# We choose the ICBM 2009c on linear asymmetric space,
# and then request the template `siibra` linked to it.
# As expected, the template is an object of type `Volume`.
icbm_tpl = siibra.spaces.get('icbm 2009c nonl asym').get_template()
icbm_tpl

# %%
# We can now fetch data from the template.
# By default (and if available), this gives us a 3D image
# in the form of a Nifti1Image object
# as defined and supported by the commonly used
# `nibabel <https://nipy.org/nibabel/index.html>`_
# library.
icbm_img = icbm_tpl.fetch()
print(type(icbm_img))

# %%
# We can  display this template with common neuroimaging visualization tools.
# Here we use the plotting tools provided by `nilearn <https://nilearn.github.io>`_
plotting.view_img(icbm_img, bg_img=None, cmap='gray', colorbar=False)

# %%
# As described above however, the template has multiple volume providers, representing different
# resources and formats. The Volume object has a list of accepted format specifiers:
icbm_tpl.formats

# %%
# Although the particular source format is usually not of interest to us,
# we want to distinguish image and mesh representations of course.
# We can use the `format` parameter of `fetch()` to specify "mesh" or "image",
# or to fetch from a concrete resource format.
# Meshes are provided as a dictionary with an Nx3 array of N vertices,
# and an Mx3 array of M triangles defined from the vertices.
# we can pre-check whether a volume provides image or mesh data
# explicitly using `provides_mesh` and `provides_image`
assert icbm_tpl.provides_mesh
icbm_mesh = icbm_tpl.fetch(format='mesh')
print(type(icbm_mesh))

# %%
# We can likewise visualize the mesh using
# plotting functions of `nilearn <https://nilearn.github.io>`_
plotting.view_surf(
    surf_mesh=[icbm_mesh['verts'], icbm_mesh['faces']], colorbar=False
)

# %%
# Some volumes are split into fragments. When fetching them,
# siibra merges these fragments into a single
# data structure, which also happened for the template mesh.
# We can also fetch individual fragments individually.
# Available fragment names are displayed when fetching,
# but we can also request an overview from the template volume.
# For fetching fragments, it is sufficient to use descriptive
# substrings.
print(icbm_tpl.fragments)
icbm_mesh_r = icbm_tpl.fetch(format='mesh', fragment='right')
plotting.view_surf(
    surf_mesh=[icbm_mesh_r['verts'], icbm_mesh_r['faces']], colorbar=False
)

# %%
# For convenience, templates may also be requested from an atlas,
# or right away from the siibra package.
mni152tpl = siibra.atlases.get('human').get_template(space="mni152")
mni152tpl = siibra.get_template("icbm 152 asym")
