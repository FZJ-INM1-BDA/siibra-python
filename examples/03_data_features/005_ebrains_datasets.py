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
.. _ebrains_datasets:

EBRAINS regional datasets
~~~~~~~~~~~~~~~~~~~~~~~~~ 

The modality "EbrainsRegionalDataset' is different from the others.
It returns any datasets in the EBRAINS Knowledge Graph which could be linked to the given atlas concept, and provides access to their metadata, as well as the link to the EBRAINS Knowledge Graph.
The returned features can thus actually have different modalities, and might include datasets which are also supported as one of the explicitly supported modalities shown in previous examples.
"""

# %%
# We query regional features for the secondary visual cortex V2.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
region = atlas.get_region("hoc2")
features = siibra.features.get(region, siibra.features.dataset.EbrainsDataFeature)
for feature in features:
    print(f" - {feature.name}")
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/ebrains_logo.png'


# %%
# Each EBRAINS feature provides access to the metadata from the EBRAINS Knowledge Graph.
# We view some of those for the last returned feature (which is accessible from the loop above).
# ``siibra`` implements a lazy loading mechanism here again:
# Once we access attributes which require deeper metadata, it will run a query to the Knowledge Graph to fetch it.
print(feature.name)
print(feature.description)

# %%
# We can use the url of the feature to access their full information in the
# Knowledge Graph. Just click on the link to test it.
print(feature.url)
