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
Sampling high-resolution image patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

# %%
from nilearn import plotting
import siibra

# %%
# Select a brain region map for searching image data.
hem = 'right'
parc = siibra.get_parcellation('julich 3.1')
region = parc.get_region('4p ' + hem)
pmap = region.extract_map('mni 152', 'statistical', name='207')

# %%
# Find relevant cellbody stained brain sections.
features = siibra.find_features(
    region,
    siibra.modality_vocab.modality.CELL_BODY_STAINING
)
print(features)
section = features[20]


# %%
imgplane = siibra.attributes.locations.plane.from_image(section)
lmap = siibra.get_map('cortical layers', space='bigbrain')
l4 = lmap.parcellation.get_region('4 ' + hemisphere)
contour = imgplane.intersect_mesh(lmap.fetch(l4, format='mesh'))
# %%
