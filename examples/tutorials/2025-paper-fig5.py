# Copyright 2018-2025
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
Anatomically guided reproducible extraction of full resolution image data from cloud resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` allows to implement reproducible workflows for sampling microscopy data from anatomically defined regions of interest.
This example retrieves the probabilistic map of motor area 4p from the Julich-Brain atlas in the right hemisphere,
uses it to find relevant high-resolution scans of whole-brain tissue sections in BigBrain space (B; ll. 8-10),
and samples oriented cortical patches centered on the cortical mid surface.
The different coordinate systems are automatically handled using precomputed nonlinear transformations.
To define the oriented cortical image patch, `siibra` intersects cortical layer surface meshes (Wagstyl. et al, PLoS biology 2020)
with BigBrain 1 micron sections (Schiffer et al. 2022; https://doi.org/10.25493/JWTF-PAB),
and finds points on the cortical mid surface with significant relevance according to the probability map of the brain area
The scoring also uses nonlinear transformations to resolve the coordinate system mismatch.
The extracted mid surface points are then used to extract the closest 3D cortical profiles from the cortical layer
maps, which provide information about orientation and thickness of the cortex at the chosen
position.
The profile is projected to the image plane of the respective 1 micron sections and used to fetch
the full resolution image data for the identified cortical patch from the underlying cloud resource.
"""

# %%
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import siibra

assert siibra.__version__ >= "1.0.1"

# %%
# First we retrieve the probability map of a motor area
# from the Julich-Brain cytoarchitectonic atlas.
region = siibra.get_region("julich 3.0.3", "4p right")
region_map = region.get_regional_map("mni152", "statistical")

# %%
# We can use the probability map as a query to extract 1 micron resolution
# cortical image patches from BigBrain.
# The patches are automatically selected to be centered on the cortex
# and cover its whole thickness.
patches = siibra.features.get(region_map, "BigBrain1MicronPatch", lower_threshold=0.52)

# For this example, we filter the results to a particular (nice) brain section,
# but the tutorial can be run without this filter.
patches = [p for p in patches if p.bigbrain_section == 3556]

# Results are sorted by relevance to the query, so in our case the first
# in the list will be the one with highest probability as defined in the
# probability map.
patch = patches[0]

# siibra samples the patch in upright position, but keeps its
# original orientation in the affine transformation encoded in the NIfTI.
# Let's first plot the pure voxel data of the patch to see that.
plt.imshow(
    patch.fetch().get_fdata().squeeze(), cmap='gray'
)

# %%
# To understand where and how siibra actually sampled this patch,
# we first plot the position of the chosen brain section in MNI space.
view = plotting.plot_glass_brain(region_map.fetch(), cmap='viridis')
roi_mni = patch.get_boundingbox().warp('mni152')
_, key, pos = min(zip(roi_mni.shape, view.axes.keys(), roi_mni.center))
view.axes[key].ax.axvline(pos, color='red', linestyle='--', linewidth=2)

# %%
# Next we plot the section itself and identify the larger region of
# interest around the patch, using the bounding box
# of the centers of most relevant patches with a bit of padding.
patch_locations = siibra.PointCloud.union(*[p.get_boundingbox().center for p in patches])
roi = patch_locations.boundingbox.zoom(1.5)

# fetch the section at reduced resolution for plotting.
section_img = patch.section.fetch(resolution_mm=0.2)
display = plotting.plot_img(
    section_img, display_mode="y", cmap='gray', annotate=False,
)
display.title(f"BigBrain section #{patch.bigbrain_section}", size=8)

# Annotate the region of interest bounding box
ax = list(display.axes.values())[0].ax
(x, w), _, (z, d) = zip(roi.minpoint, roi.shape)
ax.add_patch(plt.Rectangle((x, z), w, d, ec='k', fc='none', lw=1))


# %%
# Zoom in to the region of interest, and show the cortical layer boundaries
# that were used to define the patch dimensions.

# Since the patch locations only formed a flat bounding box,
# we first extend the roi to the patch's shape along the flat dimension.
for dim, size in enumerate(roi.shape):
    if size == 0:
        roi.minpoint[dim] = patch.get_boundingbox().minpoint[dim]
        roi.maxpoint[dim] = patch.get_boundingbox().maxpoint[dim]

# Fetch the region of interest from the section, and plot it.
roi_img = patch.section.fetch(voi=roi, resolution_mm=-1)
display = plotting.plot_img(roi_img, display_mode="y", cmap='gray', annotate=False)
ax = list(display.axes.values())[0].ax

# Intersect cortical layer surfaces with the image plane
plane = siibra.Plane.from_image(patch)
layermap = siibra.get_map("cortical layers", space="bigbrain")
layer_contours = {
    layername: plane.intersect_mesh(layermap.fetch(region=layername, format="mesh"))
    for layername in layermap.regions
}

# Plot the contours on top of the image, using the
# colormap provided by siibra.
for layername, contours in layer_contours.items():
    layercolor = layermap.get_colormap().colors[layermap.get_index(layername).label]
    for contour in contours:
        for segment in contour.crop(roi):
            X, _, Z = segment.coordinates.T
            ax.plot(X, Z, "-", ms=4, color=layercolor)

# %%
# Plot the region of interest again, this time with the cortical profile that
# defined the patch, as well as other candidate patch's locations
# with their relevance scores, ie. probabilities.
display = plotting.plot_img(roi_img, display_mode="y", cmap='gray', annotate=False)
ax = list(display.axes.values())[0].ax

# Concatenate all coordinates of the layer 4 intersected contours.
layer = "cortical layer 4 right"
XYZ = np.vstack([c.coordinates for c in layer_contours[layer]])
layerpoints = siibra.PointCloud(XYZ, space='bigbrain')
patch_probs = region_map.evaluate_points(layerpoints)
X, _, Z = layerpoints.coordinates.T
ax.scatter(X, Z, c=patch_probs, s=10)

# plot the cortical profile in red
X, _, Z = patch.profile.coordinates.T
ax.plot(X, Z, "r-", lw=2)

# sphinx_gallery_thumbnail_number = -2
