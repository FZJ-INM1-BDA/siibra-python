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
Access BigBrain high-resolution data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

siibra provides access to high-resolution image data parcellation maps defined
for the 20 micrometer BigBrain space.
The BigBrain is very different from other templates. Its native resolution is
20 micrometer, resulting in about one Terabyte of image data. Yet, fetching the
template works the same way as for the MNI templates, with the difference that
we can specify a reduced resolution or volume of interest to fetch a feasible
amount of image data, or a volume of interest.
"""

# %%
# We start by selecting an atlas.
import siibra
from nilearn import plotting
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/02-004.png'

# %%
# Per default, `siibra` will fetch the whole brain volume at a reasonably
# reduced resolution.
space = siibra.spaces['bigbrain']
bigbrain_template = space.get_template()
bigbrain_whole_img = bigbrain_template.fetch()
try:
    plotting.view_img(bigbrain_whole_img, bg_img=None, cmap='gray')
except:
    print(type(bigbrain_whole_img))

# %%
# To see the full resolution, we may specify a bounding box in the physical
# space. You will learn more about spatial primitives like points and bounding
# boxes in :ref:`locations`. For now, we just define a volume of interest from
# two corner points in the histological space. We specify the points with
# a string representation, which could be conveniently copy pasted from the
# interactive viewer `siibra explorer <https://atlases.ebrains.eu/viewer>`_.
# Note that the coordinates can be specified by 3-tuples, and in other ways.
voi = siibra.locations.BoundingBox(
    point1="-30.590mm, 3.270mm, 47.814mm",
    point2="-26.557mm, 6.277mm, 50.631mm",
    space=space
)
bigbrain_chunk = bigbrain_template.fetch(voi=voi, resolution_mm=0.02)
plotting.view_img(bigbrain_chunk, bg_img=None, cmap='gray')

# %%
# Note that since both fetched image volumes are spatial images with a properly
# defined transformation between their voxel and physical spaces, we can
# directly plot them correctly superimposed on each other:
plotting.view_img(
    bigbrain_chunk,
    bg_img=bigbrain_whole_img,
    cmap='magma',
    cut_coords=tuple(voi.center)
)

# %%
# Next we select a parcellation which provides a map for BigBrain, and extract
# labels for the same volume of interest. We choose the cortical layer maps by `Wagstyl et al<https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678>`.
# Note that by specifying "-1" as a resolution, `siibra` will fetch the highest
# possible resolution.
layermap = siibra.get_map(space='bigbrain', parcellation='layers')
mask = layermap.fetch(fragment='left hemisphere', resolution_mm=-1, voi=voi)
mask

# %%
# Since we operate in physical coordinates, we can plot both image chunks
# superimposed, even if their resolution is not exactly identical.
plotting.view_img(mask, bg_img=bigbrain_chunk, opacity=.2, symmetric_cmap=False)

# %%
# `siibra` can help us to assign a brain region to the position of the volume
# of interest. This is covered in more detail in :ref:`assignment`. For now,
# just note that `siibra` can employ spatial objects from different template spaces.
# Here it automatically warps the centroid of the volume of interest to MNI space
# for location assignment.
julich_pmaps = siibra.get_map(space='mni152', parcellation='julich', maptype='statistical')
assignments = julich_pmaps.assign(voi.center)
assignments

# %%
# 1 micron scans of BigBrain sections across the brain can be found as
# VolumeOfInterest features. The result is a high-resolution image structure,
# just like the bigbrain template.
hoc5l = siibra.get_region('julich 2.9', 'hoc5 left')
features = siibra.features.get(
    hoc5l,
    siibra.features.cellular.CellbodyStainedSection
)
# let's see the names of the found features
for f in features:
    print(f.name)

# %%
# Now fetch the 1 micron section at a lower resolution, and display in 3D space.
section1402 = features[3]
plotting.plot_img(
    section1402.fetch(),
    bg_img=bigbrain_whole_img,
    title="#1402",
    cmap='gray'
)

# %%
# Let's fetch a crop inside hoc5 at full resolution. We intersect the bounding
# box of hoc5l and the section.
hoc5_bbox = hoc5l.get_bounding_box('bigbrain').intersection(section1402.boundingbox)
print(f"Size of the bounding box: {hoc5_bbox.shape}")

# this is quite large, so we shrink it
voi = hoc5_bbox.zoom(0.1)
crop = section1402.fetch(voi=voi, resolution_mm=-1)
plotting.plot_img(crop, bg_img=None, cmap='gray')
