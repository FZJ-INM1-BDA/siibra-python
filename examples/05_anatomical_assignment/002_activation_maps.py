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

Continuous parcellations maps can also assign regions to image volumes. 
If given a ``Nifti1Image`` object as input, the assignment method will interpret it as a measurement of a spatial distribution.
It will first split the image volume into disconnected components, i.e. any subvolumes which are clearly separated by zeros.
Then, each component will be compared to each continuous maps in the same way that the Gaussian blobs representing uncertain points are processed in 
:ref:`sphx_glr_examples_05_anatomical_assignment_001_coordinates.py`.
"""
# %%
# We start again by selecting the Julich-Brain probabilistic maps from the human atlas, which we will use for the assignment. 
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
with siibra.QUIET: # suppress progress output here
    julich_pmaps = atlas.get_map(
        space="mni152",
        parcellation="julich",
        maptype="continuous"
    )

# %%
# As an exemplary input signal, we threshold a continuous map from the
# functional maps (DiFuMo 64) by Thirion et al.
with siibra.QUIET: # suppress progress output here
    difumo_maps = atlas.get_map(
        space='mni152',
        parcellation='difumo 64', 
        maptype='continuous'
    )
from nilearn import image
img = image.math_img( 
    "im1+im2",
    im1 = image.threshold_img(difumo_maps.fetch(15), 0.0003),
    im2 = image.threshold_img(difumo_maps.fetch(44), 0.0003)
)
region1 = difumo_maps.decode_label(15)
region2 = difumo_maps.decode_label(44)

# let's look at the resulting image
from nilearn import plotting
plotting.view_img(img, 
    title="Thresholded functional map",
    colorbar=False
)

# %%
# We now assign Julich-Brain regions to this functional map with multiple
# modes. Note that when given an image as input, the ``assign`` method
# will output an image volume with labels for the detecteded components,
# so we can lookup which component corresponds to which part of the 
# input volume.
# Since we are here ususally interested in correlations of the modes,
# we filter the result by significant (positive) correlations.
with siibra.QUIET: # suppress progress output
    assignments, components = julich_pmaps.assign(img)
print(f"DiFuMo regions used: {region1.name}, {region2.name}")
assignments[assignments.Correlation>=0.2]


