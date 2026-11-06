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
.. _001_coordinates
:bdg-secondary:`Intermediate`

Assigning coordinates to brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` can use statistical parcellation maps to assign exact and uncertain
coordinates to brain regions. We start by selecting the Julich-Brain
probability maps of the human atlas, which we will use for the assignment.
"""

# %%
import siibra
from nilearn import plotting

# %%
# Choose a parcellation map. This example uses probability maps.
with siibra.QUIET:  # suppress progress output
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9",
        space="mni152",
        maptype="statistical"
    )

# %%
# **Assigning an exact coordinate.**
# We assign a point in MNI reference space, chosen to lie in the postcentral
# gyrus (PostCG) of the right hemisphere. For more information about
# ``siibra.Point``, see :ref:`locations`.
#
# For an exact coordinate, ``assign`` reads each probability map at the voxel
# containing the point. The result is a pandas DataFrame with one row per
# region whose probability at the point is above zero:
#
# - ``region``: the brain region.
# - ``map value``: the probability of the region at the point.
# - ``input structure``: the index of the point in the input, here 0.
# - ``centroid``: the coordinate of the point.
#
# Sorting by ``map value`` puts the most probable region first, which is
# indeed in PostCG.
point = siibra.Point((27.75, -32.0, 63.725), space='mni152')
with siibra.QUIET:  # suppress progress output
    assignments = julich_pmaps.assign(point)
assignments.sort_values(by=['map value'], ascending=False)

# %%
# **Assigning a coordinate with location uncertainty.**
# Coordinates are rarely exact. A position obtained from an sEEG electrode, for
# example, is uncertain by several millimeters, and more so after warping it to
# a reference space. We therefore give the point an uncertainty with
# ``sigma_mm``.
#
# If the uncertainty is at least 3 voxels of the map (3 mm here), ``assign``
# does not read single voxels. It represents the point as a 3D Gaussian kernel
# with that standard deviation, extending 3 standard deviations around the
# point, and compares the kernel with each probability map, as it compares
# image inputs. Each row then holds these scores:
#
# - ``map value``: the highest probability of the region within the kernel.
# - ``correlation``: Pearson correlation coefficient between the kernel and the
#   probability map, over the bounding box of both.
# - ``intersection over union``: the voxels in both the kernel and the
#   probability map, relative to the voxels in either.
# - ``map containedness``: the fraction of the probability map's voxels that
#   lie within the kernel.
# - ``input containedness``: the fraction of the kernel's voxels that lie
#   within the probability map.
# - ``map weighted mean``: the mean probability of the region, weighted by the
#   kernel.
# - ``input weighted mean``: the mean kernel value, weighted by the probability
#   map.
#
# The intersection and containedness scores count the voxels above zero,
# regardless of their values; the correlation and the weighted means use the
# values.
#
# A high ``input containedness`` means that the point most likely lies in that
# region. We keep the regions that contain at least half of the kernel.
point_uncertain = siibra.Point((27.75, -32.0, 63.725), space='mni152', sigma_mm=3.)
with siibra.QUIET:  # suppress progress output
    assignments = julich_pmaps.assign(point_uncertain)
assignments.query('`input containedness` >= 0.5').dropna(axis=1)

# %%
# To verify the result, we plot the probability maps of these regions at the
# point.
for index, assignment in assignments[assignments['input containedness'] >= 0.5].iterrows():
    pmap = julich_pmaps.fetch(region=assignment['region'])
    plotting.plot_stat_map(pmap, cut_coords=tuple(point), title=assignment['region'].name, cmap='magma')
