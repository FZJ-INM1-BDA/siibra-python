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
Accessing parcellation maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just like reference templates (covered in :ref:`templates`), a parcellation map
or region map is a spatial object - usually a volumetric image with discrete
voxel labels for each region, a binary mask image, or a 3D distribution of
float values in case of a probabilistic map. Since some parcellations provide maps
in different reference spaces, obtaining a parcellation map involves to specify
a reference space.
"""

# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS # siibra.atlases["human"]

# for plotting
from nilearn import plotting
# (in order to display plots using web browsers) 
import matplotlib
matplotlib.use('webagg')

# %%
# We select the maximum probability map of Julich-Brain in MNI152 space,
# which is a parcellation map with discrete labels. The `get_map` method
# of the atlas assumes maptype='labelled' by default.
julich_mpm = atlas.get_map(space="mni152", parcellation="julich", maptype="labelled")
print(julich_mpm)

# %%
# Some maps are shopeed with several volumes inside. Julich-Brain ships the left
# and right hemispheres in different volumes, so corresponding regions can use
# the same label index while still being distinguishable. We can find the number
# of mapped volumes inside a parcellation map simply from its length:
print(len(julich_mpm))
# and list the volumes
[v for v in julich_mpm]

# As already seen for reference templates, the returned object does not contain
# any image data yet, since siibra uses a lazy strategy for loading data.
# To access the actual image data, we call the fetch() method.
# This gives us a Nifti1Image object defined in the reference space.
mapimg = julich_mpm.fetch(volume=0) # need to specify which volume is fetched 
print(type(mapimg))

# we can now plot the image with nilearn
plotting.plot_roi(mapimg, title=f"{julich_mpm.parcellation.name}")

# %%
# To get the second labelled volume, we can pass volume=1 or instead of
# dealing with map indices, we can simply iterate over all available maps
# using `fetch_iter()`.
for img in julich_mpm.fetch_iter():
    plotting.plot_roi(img)

# %%
# Julich-Brain, like some other parcellations, provides probabilistic maps.
# The maximum probability map is just a simplified representation, displaying
# for each voxel the label of the brain region with highest probability. We can
# access the probabilistic information by requesting the
# "continuous" maptype (`siibra.maptype.CONTINUOUS`).
# Note that since the set of probability maps are usually a large number of
# sparsely populated image volumes, `siibra` will load the volumetric data only
# once and then convert it to a sparse index format, that is much more
# efficient to process and store. The sparse index is cached on the local disk,
# therefore subsequent use of probability maps will be much faster.
with siibra.QUIET:  # suppress progress output
    julich_pmaps = atlas.get_map(space="mni152", parcellation="julich", maptype="continuous")
julich_pmaps

# %%
# Since the continuous maps overlap, this map provides access to several
# hundreds of brain volumes.
print(len(julich_pmaps))

# %%
# Again, we can iterate over all pmaps using `fetch_iter()`,
# but for simplicity, we just access a random index here.
pmap = julich_pmaps.fetch(volume=102)
plotting.plot_stat_map(pmap)

# %%
# Now we might wonder which region this map index actually refers to.
# The Parcellation map objects can safely translate
# indices into regions and vice versa:

# What is the region behind map index 102?
julich_pmaps.get_region(vol=102)

# Or vice versa, what is the index of that region?
julich_pmaps.get_index('IFJ1 left')

# %%
# In addition to parcellation maps, `siibra` can produce binary masks of brain regions.
hoc5L = atlas.get_region('hoc5 left', parcellation='julich 2.9')
hoc5L_mask = hoc5L.build_mask(space="mni152", maptype="continuous")
plotting.plot_roi(hoc5L_mask, title=f"Mask of the region {hoc5L.name}")
