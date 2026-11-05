# Copyright 2018-2026
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
.. _002_activation_maps
:bdg-secondary:`Intermediate`

Assign modes in activation maps to brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistical parcellation maps can also assign brain regions to image volumes,
such as activation maps. ``assign`` treats the image as a spatial
distribution. It first splits the image into connected components, i.e. parts
separated by zeros, and skips components of a single voxel. Then it compares
each component with each probability map, using the scores described for
uncertain points in
:ref:`sphx_glr_examples_05_anatomical_assignment_001_coordinates.py`.

We start again by selecting the Julich-Brain probability maps of the human
atlas, which we will use for the assignment.
"""


# %%
import siibra
from nilearn import plotting
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/05-002.png'

# %%
# Select the probability maps for the assignment.
julich_pmaps = siibra.get_map(
    parcellation="julich 2.9",
    space="mni152",
    maptype="statistical"
)

# %%
# As an example input, we use a statistical map from the 64-component
# functional mode parcellation (DiFuMo 64) by Thirion et al.
difumo_maps = siibra.get_map(
    parcellation='difumo 64',
    space='mni152',
    maptype='statistical'
)
region = "fusiform posterior"
img = difumo_maps.fetch(region=region)

# let's look at the resulting query image
plotting.view_img(
    img,
    symmetric_cmap=False,
    cmap="magma"
)

# %%
# This map has two modes, one in each hemisphere. To assign it, we first wrap
# the image in a ``Volume`` with its reference space.
#
# Each row of the result pairs one component with one region:
#
# - ``input structure``: the label of the component, numbered from 1.
# - ``centroid``: the center of the component, in physical coordinates.
# - ``map value``: the highest probability of the region within the component.
# - ``correlation``, ``intersection over union``, ``map containedness``,
#   ``input containedness``, ``map weighted mean``, ``input weighted mean``:
#   the scores described for uncertain points.
#
# Rows are ordered by component, then by region. To compare the image as a
# whole instead of component by component, pass ``split_components=False``.
#
# Here we look for regions whose probability map resembles one of the modes,
# so we keep the rows with a correlation of at least 0.35.
volume = siibra.volumes.volume.from_nifti(img, difumo_maps.space, "fusiform posterior")
with siibra.QUIET:  # suppress progress output
    assignments = julich_pmaps.assign(volume)
assignments.query('correlation >= 0.35')
