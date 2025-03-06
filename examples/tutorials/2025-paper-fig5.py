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
import matplotlib.pyplot as plt
import numpy as np
import siibra

assert siibra.__version__ >= "1.0.1"

# %%
# 1: Retrieve probability map of a motor area in Julich-Brain
region = siibra.get_region("julich 3.0.3", "4p right")
region_map = siibra.get_map("julich 3.0.3", "mni152", "statistical").get_volume(region)

# %%
# 2: Extract BigBrain 1 micron patches with high probability in this area
patches = siibra.features.get(region_map, "BigBrain1MicronPatch", lower_threshold=0.5)
print(f"Found {len(patches)} patches.")

# %%
# 3: Display highly rated samples, here further reduced to a predefined section
section_num = 3556
candidates = [p for p in patches if p.bigbrain_section == section_num]
patch = candidates[0]
plt.figure()
plt.imshow(patch.fetch().get_fdata().squeeze(), cmap="gray", vmin=0, vmax=2**16)
plt.axis("off")
plt.title(f"#{section_num} - {patch.vertex}", fontsize=10)

# %%
# To understand how the live query works, we have a look at some of the intermediate
# steps that `siibra` is performing under the hood. It first identifies brain sections that
# intersect the given map (or, more generally, the given image volume).
# Each returned patch still has the corresponding section linked, so we can have a look at it.
# The section was intersected with the cortical layer 4 surface to get an approximation of
# the mid cortex. This can be done by fetching the layer surface meshes, and intersecting
# them with the 3D plane corresponding to the brain section.
plane = siibra.Plane.from_image(patch)
layermap = siibra.get_map("cortical layers", space="bigbrain")
layer_contours = {
    layername: plane.intersect_mesh(layermap.fetch(region=layername, format="mesh"))
    for layername in layermap.regions
}
ymin, ymax = [p[1] for p in patch.section.get_boundingbox()]
crop_voi = siibra.BoundingBox((17.14, ymin, 40.11), (22.82, ymax, 32.91), 'bigbrain')
cropped_img = patch.section.fetch(voi=crop_voi, resolution_mm=-1)
phys2pix = np.linalg.inv(cropped_img.affine)

# The probabilities can be assigned to the contour vertices with the
# probability map.
points = siibra.PointCloud.union(
    *[c.intersection(crop_voi) for c in layer_contours["cortical layer 4 right"]]
)
# siibra warps points to MNI152 and reads corresponding PMAP values
probs = region_map.evaluate_points(points)
img_arr = cropped_img.get_fdata().squeeze().swapaxes(0, 1)
plt.imshow(img_arr, cmap="gray", origin="lower")
X, Y, Z = points.transform(phys2pix).coordinates.T
plt.scatter(X, Z, s=10, c=probs)
prof_x, _, prof_z = zip(*[p.transform(phys2pix) for p in patch.profile])
plt.plot(prof_x, prof_z, "r", lw=2)
plt.axis("off")

# %%
# We can plot the contours on top of the image, and even use the
# colormap recommended by siibra. We use a crop around the brain
# region to zoom a bit closer to the extracte profile and patch.
plt.figure()
plt.imshow(img_arr, cmap="gray", vmin=0, vmax=2**16, origin="lower")
for layername, contours in layer_contours.items():
    layercolor = layermap.get_colormap().colors[layermap.get_index(layername).label]
    for contour in contours:
        for segment in contour.crop(crop_voi):
            pixels = segment.transform(phys2pix, space=None).homogeneous
            plt.plot(pixels[:, 0], pixels[:, 2], "-", ms=4, color=layercolor)

# plot the profile
plt.plot(prof_x, prof_z, "r", lw=2)
plt.axis("off")

# sphinx_gallery_thumbnail_number = -2
