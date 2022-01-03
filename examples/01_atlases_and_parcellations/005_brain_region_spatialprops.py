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
Spatial properties of brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a reference space is specified, brain regions can expose a range of
spatial properties.
"""

# %%
# We start by selecting an atlas and a space
from pprint import pprint # we'll use that for some outputs
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
space = atlas.spaces.MNI152_2009C_NONL_ASYM

# %%
# The `Region.spatial_props()` method computes the centroid and volume of a
# brain region.  Note that a brain region
# might in general consist of multiple separated components in the space.
# Also note that in `siibra`, spatial properties are always represented in
# millimeter units of the physical coordinate system of the reference space,
# not in voxel units.
v1_left = atlas.get_region("v1 left", parcellation="2.9 152")
props = v1_left.spatial_props(space)
pprint(props)

# %%
# The returned centroid is `siibra.Point` object. Such spatial primitives are 
# covered in more detail in :ref:`locations`. For now, we just acknowledge
# that this minimizes misinterpretation of the coordinates, since a siibra
# Point is explicitely linked to its space.
centroid = props['components'][0]['centroid']
print(centroid)
centroid.space.name

# %% 
# We can also generate a binary mask of the region in a given space, which
# gives us a Nifti1Image object as provided by `nibabel <https://nipy.org/nibabel/>`_, 
# and which we can directly visualize using plotting functions like the ones in 
# `nilearn <https://nilearn.github.io/stable/index.html>`_:
mask = v1_left.build_mask(space)
from nilearn import plotting
plotting.plot_roi(mask, title=f"Mask of {v1_left.name} in {space.name}")

