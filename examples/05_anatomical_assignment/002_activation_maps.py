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
Assign modes in activation maps to brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistical parcellations maps can also assign regions to image volumes.
If given a ``Nifti1Image`` object as input, the assignment method will interpret it as a measurement of a spatial distribution.
It will first split the image volume into disconnected components, i.e. any subvolumes which are clearly separated by zeros.
Then, each component will be compared to each statistical maps in the same way that the Gaussian blobs representing uncertain points
are processed in :ref:`sphx_glr_examples_05_anatomical_assignment_001_coordinates.py`.

We start again by selecting the Julich-Brain probabilistic maps from the human atlas, which we will use for the assignment.
"""


# %%
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/05-002.png'
import siibra
from nilearn import plotting

# %%
# Select a probabilistic parcellation map to do the anatomical assignments.
julich_pmaps = siibra.get_map(
    parcellation="julich 2.9",
    space="mni152",
    maptype="statistical"
)

# %%
# As an exemplary input signal, we use a 
# statistical map from the 64-component functional mode
# parcellation (DiFuMo 64) by Thirion et al.
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
    title=f"Functional map created from {region}",
    symmetric_cmap=False,
    colorbar="south"
)

# %%
# This "fake functional map" has two modes, one in each hemisphere.
# We now assign cytoarchitectonic regions to this functional map.
# Since we are here usually interested in correlations of the modes,
# we filter the result by significant (positive) correlations.
with siibra.QUIET:  # suppress progress output
    assignments = julich_pmaps.assign(img)
assignments.query('Correlation >= 0.35')


# %%
