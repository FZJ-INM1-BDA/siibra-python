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

# %%
# 1: Retrieve probability map of a motor area in Julich-Brain
parc = siibra.parcellations.get('julich 3.1')
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
f, axs = plt.subplots(1, 3, figsize=(8, 24))
for patch, ax in zip(list(candidates)[:3], axs.ravel()):
    patchdata = patch.fetch().get_fdata().squeeze()
    ax.imshow(patchdata, cmap='gray', vmin=0, vmax=2**16)
    ax.axis('off')
    ax.set_title(f"#{section} - {patch.vertex}", fontsize=10)

