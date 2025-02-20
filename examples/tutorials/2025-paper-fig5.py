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
import siibra
assert siibra.__version__ >= "1.0.1"
import matplotlib.pyplot as plt
import numpy as np

# %%
# 1: Retrieve probability map of a motor area in Julich-Brain
parc = siibra.parcellations.get('julich 3.0.3')
region = parc.get_region("4p right")
pmap = parc.get_map('mni152', 'statistical').get_volume(region)

# %%
# 2: Extract BigBrain 1 micron patches with high probability in this area
patches = siibra.features.get(pmap, "BigBrain1MicronPatch", lower_threshold=0.7)
print(f"Found {len(patches)} patches.")

# %%
# 3: Display highly rated samples, here further reduced to a predefined section
section = 3556
candidates = filter(lambda p: p.bigbrain_section == 3556, patches)
patch = next(iter(candidates))
plt.figure()
patchdata = patch.fetch().get_fdata().squeeze()
plt.imshow(patchdata, cmap='gray', vmin=0, vmax=2**16)
plt.axis('off')
plt.set_title(f"#{section} - {patch.vertex}", fontsize=10)


# %%
# To understand how the live query works, we have a look at some of the intermediate
# steps that `siibra` is performing under the hood.
# It first identifies brain sections that intersect the given map (or, more generally, the given image volume.)
# Each returned patch still has the corresponding section linked, so we can have a look at it.
section = patch._section
fig = plt.figure()
ax = fig.add_subplot(111)
img = section.fetch(resolution_mm=0.8)
plt.imshow(img.get_fdata().squeeze(), cmap='gray', vmin=0, vmax=2**16)

# The section was intersected with the cortical layer 4 surface to get an approximation of the mid cortex. 
# This can be done by fetching the layer surface meshes, and intersecting 
# them with the 3D plane corresponding to the brain section.
plane = siibra.Plane.from_image(section)
layermap = siibra.get_map('cortical layers', space='bigbrain')
layer_contours = {
    layername: plane.intersect_mesh(
        layermap.fetch(region=layername, format='mesh')
    )
    for layername in layermap.regions
}

# The probabilities can be assigned to the countour vertices with the
# probability map.
points = siibra.PointCloud(
    np.vstack([
        contour.coordinates 
        for contour in layer_contours["cortical layer 4 right"]
    ]),
    space='bigbrain'
)
probs = pmap.evaluate_points(points)
X, Y, Z = points.transform(np.linalg.inv(pmap.affine), space=None).coordinates.T
ax.scatter(X, Z, c=probs)

# %%
# we can plot the contours on top of the image, and even use the
# colormap recommended by siibra. We use a crop around the brain
# region to zoom a bit closer to the extracte profile and patch.
crop_voi = (
    section
    .get_boundingbox()
    .intersection(pmap.get_boundingbox().zoom(0.4))
)
crop = section.fetch(voi=crop_voi, resolution_mm=0.2)
phys2pix = np.linalg.inv(crop.affine)




# plot the contour segments
plt.figure()
img = section.fetch(resolution_mm=0.8)
plt.imshow(crop.get_fdata().squeeze(), cmap='gray', vmin=0, vmax=2**16)
for layername, contours in layer_contours.items():
    for contour in contours:
        for segment in contour.crop(crop_voi):
            pixels = segment.transform(phys2pix, space=None).homogeneous
            plt.plot(
                pixels[:, 2], pixels[:, 0], '-', ms=4,
                color=layercolors[layermap.get_index(layername).label]
            )

# plot the profile points
for p in patch._profile:
    x, y, z = p.transform(phys2pix, space=None)
    plt.plot(z, x, 'r.', ms=3)
    
plt.axis('off')
# %%
