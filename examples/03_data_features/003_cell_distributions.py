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

Another regional data feature are cortical distributions of cell bodies.
The distributions are measured crom cortical image patches that have been extracted
from the original cell-body stained histological sections of the Bigbrain (Amunts et al., 2013),
scanned at 1 micrometer resolution. These patches come together with manual annotations of cortical layers.
The cell segmentations have been performed using the recently proposed Contour Proposal Networks (CPN; E. Upschulte et al.;
https://arxiv.org/abs/2104.03393; https://github.com/FZJ-INM1-BDA/celldetection).
"""


# %%
import siibra
import matplotlib.pyplot as plt
from nilearn import plotting
# sphinx_gallery_thumbnail_number = 1

# %%
# Find cell density profiles for V1. Cortical profile features are combined
# together as elements to form :ref:`CompoundFeatures<compoundfeatures>`.
# Therefore, we can select the first and only item in the results.
v1 = siibra.get_region("julich 3", "v1")
cf = siibra.features.get(v1, siibra.features.cellular.CellDensityProfile)[0]
print(cf.name)

# %%
# We can browse through the elements with integer index. To illustrate, let us
# look at the default visualization the first of them, this time using `plotly`
# backend. This will actually fetch the cell segmentation data.
print(cf[0].name)
cf[0].plot(backend="plotly")

# %%
# The segmented cells are stored in each feature as a numpy array with named columns.
cells = cf[0].cells
print("Number of segmented cells:", len(cells))
cells.head()

# %%
# We can, for example, plot the 2D distribution of the cell locations colored by layers:
plt.scatter(cells.x, cells.y, c=cells.layer, s=0.2)
plt.title(f"Cell distributions in {v1.name}")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()

# %%
# Having the data in data frame format allows further flexibility such as:
layer1_cells = cells.query('layer == 1')
plt.scatter(
    layer1_cells.x, layer1_cells.y,
    s=layer1_cells["area(micron**2)"], c=layer1_cells.label
)
area_layer1 = layer1_cells["area(micron**2)"]
plt.title(f"Mean cell area in layer 1: {area_layer1.mean()}")

# %%
# The features also have location information. We can plot their location in
# BigBrain space:
location = cf[0].anchor.location
print(location)

# fetch the template of the location's space
template = location.space.get_template().fetch()
view = plotting.plot_anat(anat_img=template, cut_coords=tuple(location))
view.add_markers([tuple(location)])
