# Copyright 2018-2024
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
Fetching custom maps and templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides the utility for developers to provide custom reference spaces
and maps.

Below is an example of the integration with [BrainGlobeAPI](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html).
It hopefully demonstrates that incorporating with existing APIs can be done with ease.

"""

# %%
# List the supported atlases in brainglobe API
import siibra
from siibra.retrieval_new.api_fetcher import brainglobe
brainglobe.ls()

# %%
# Specify that we will import the atlas with the name `allen_human_500um_v0.1`
# the `use` method returns the space, parcellation and parcellationmap
# 
space, parcellation, parcellationmap = brainglobe.use("allen_human_500um_v0.1")

# %%
# The `use` method also registers it to siibra, so we can find it using siibra methods
assert siibra.get_parcellation("allen human") is parcellation

