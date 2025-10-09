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
BigBrain PLI images
~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
bigbrain = siibra.spaces["bigbrain"]
volumes = siibra.features.get(bigbrain, "blockface")
for v in volumes:
    print(v.name)
    print(v.modality)

# %%
blockface = [v for v in volumes if "The Enriched Connectome" in v.name][0]
plotting.plot_img(
    blockface.fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
)
