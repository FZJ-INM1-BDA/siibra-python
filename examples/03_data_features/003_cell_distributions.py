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
Cortical cell body distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another regional data feature are cortical distributions of cell bodies. The distributions are measured crom cortical image patches that have been extracted from the original cell-body stained histological sections of the Bigbrain (Amunts et al., 2013), scanned at 1 micrometer resolution. These patches come together with manual annotations of cortical layers. The cell segmentations have been performed using the recently proposed Contour Proposal Networks (CPN; E. Upschulte et al.; https://arxiv.org/abs/2104.03393; https://github.com/FZJ-INM1-BDA/celldetection).
"""


# %%
# We start by selecting an atlas.
import siibra

atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# Find cell density features for V1
v1 = atlas.get_region("v1")
features = siibra.get_features(v1, siibra.modalities.CorticalCellDistribution)
print(f"{len(features)} found for region {v1.name}")

# %%
# Look at the default visualization the first of them.
# This will actually fetch the image and cell segmentation data.
features[0].plot()

# %%
# The segmented cells are stored in each feature as a numpy array with named columns.
# For example, to plot the 2D distribution of the cell locations colored by
# layers, we can do:
import matplotlib.pyplot as plt

c = features[0].cells
plt.scatter(c["x"], c["y"], c=c["layer"], s=0.1)
plt.title(f"Cell distributions in {v1.name}")
plt.grid(True)
plt.axis("square")

# %%
# The features also have location information. We can plot their location in
# BigBrain space:
location = features[0].location

# fetch the template of the location's space
template = location.space.get_template().fetch()
from nilearn import plotting

view = plotting.plot_anat(anat_img=template, cut_coords=tuple(location))
view.add_markers([tuple(location)])
