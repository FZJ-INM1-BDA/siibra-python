# Copyright 2018-2023
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
Utilizing locations of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides common locations of interests on reference spaces as objects.
These location objects can be used to query features or assignment to the maps,
see :ref:`sphx_glr_examples_05_anatomical_assignment_001_coordinates.py` and
:ref:`sphx_glr_examples_05_anatomical_assignment_002_activation_maps.py`.
The conversion of these locations to other spaces are done in the background
but one can also be invoked when needed.
"""

# %%
import siibra
from nilearn import plotting
import numpy as np

# %%
# The simplest location type is a point. It can have a uncertainty variable
# or can be exact.
point = siibra.Point((27.75, -32.0, 63.725), space='mni152')
point_uncertain = siibra.Point((27.75, -32.0, 63.725), space='mni152', sigma_mm=3.)
point_uncertain

# %%
# We can create a PointCloud from several points or a set of coordinates.
siibra.PointCloud(
    [(27.75, -32.0, 63.725), (27.75, -32.0, 63.725)],
    space='mni152',
    sigma_mm=[0.0, 3.]
)

# %%
# There are several helper properties and methods for locations, some specific
# to the location type. For example, we can warp the points to another space
# (currently limited only between MNI152, COLIN27, and BigBrain).
print(point.warp('bigbrain'))
print(point.warp('colin27'))

# %%
# To explore further, let us first create a random pointcloud
ptset = siibra.PointCloud(
    np.array([
        np.random.randn(10000) * 3 - 27.75,
        np.random.randn(10000) * 3 - 32.0,
        np.random.randn(10000) * 3 + 63.725
    ]).T,
    space='mni152'
)

# %%
# We can display these points as a kernel density estimated volume
kde_volume = siibra.volumes.from_pointcloud(ptset)
plotting.view_img(kde_volume.fetch())

# %%
# Moreover, a location object can be used to query features. While we can also,
# query with the pointcloud, let us use the bounding box that encloses these
# points. We will query for image features and print the assignment of the
# anatomical anchors to our BoundingBox
bbox = ptset.boundingbox
features_of_interest = siibra.features.get(bbox, 'image')
for f in features_of_interest:
    print(f.last_match_description)

# %%
# And now let us simply select the features that overlaps with our BoundingBox
selected_features = [
    f
    for f in features_of_interest
    if "overlaps" in str(f.last_match_result[0].qualification)
]
for f in selected_features:
    print(f.name)
