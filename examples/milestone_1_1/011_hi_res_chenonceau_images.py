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
# Query for images on MNI 152 2009c Aym Nonl space and filter out the ones from
# Chenonceau dataset
mni152 = siibra.spaces["mni152"]
image_features = [
    f
    for f in siibra.features.get(mni152, siibra.features.generic.Image)
    if "Chenonceau" in f.name
]
for f in image_features:
    print(f.name)
    print(f.modality)

# %%
# As we see from above, there are several modalities to choose from. For
# demonstration, fetch the image for "proton density" modality and plot it
proton_density = siibra.features.get(
    mni152, siibra.features.generic.Image, modality="proton density"
)[0]
pd_img = proton_density.fetch()
plotting.plot_img(pd_img, cmap="magma", title=proton_density.name)

# %%
# The set of images integrated also contain T2-weighted anatomical images
# of the Chenonceau brain. This can be used to overlay other modalities. Since
# MNI 152 template is only in 1mm resolution, the the underlying anatomoical
# structure is not present as it would be with Chenonceau 100um or 150um
# images.
t2_weighted_150um = [
    f
    for f in siibra.features.get(
        mni152,
        siibra.features.generic.Image,
        modality="T2-weighted (T2w) image",
    )
    if "150 micrometer" in f.name
][0]
plotting.view_img(
    pd_img,
    bg_img=t2_weighted_150um.fetch(),
    cmap="magma",
    symmetric_cmap=False,
    title=f"{proton_density}\non {t2_weighted_150um}",
    vmin=1e-7,
    vmax=0.15,
    opacity=0.5,
)
