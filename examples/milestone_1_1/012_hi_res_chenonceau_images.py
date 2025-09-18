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
Hi-Res Human Brain Images - Chenonceau
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
mni152 = siibra.spaces["mni152"]
image_features = siibra.features.get(
    mni152, siibra.features.generic.Image
)
for f in image_features:
    print(f.modality)

# %%
proton_density = siibra.features.get(
    mni152, siibra.features.generic.Image, modality="proton density"
)[0]
pd_img = proton_density.fetch()
plotting.plot_img(pd_img, cmap="magma")
