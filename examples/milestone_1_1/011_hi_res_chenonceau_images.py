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
# demonstration, fetch the image for modality of fractional anisotropy and
# plot it
fractional_anisotropy = siibra.features.get(
    mni152, siibra.features.generic.Image, modality="fractional anisotropy"
)[0]
fa_img = fractional_anisotropy.fetch()
plotting.plot_img(fa_img, cmap="magma", title=fractional_anisotropy.name)

# %%
# The set of images integrated also contain T2-weighted anatomical images
# of the Chenonceau brain. This can be used to overlay other modalities. Since
# MNI 152 template is only in 1mm resolution, the the underlying anatomical
# structure is not present as it would be with Chenonceau 100um or 150um
# images. First, we plot 150um to see the details of the anatomical image
t2_weighted_150um = [
    f
    for f in siibra.features.get(
        mni152,
        siibra.features.generic.Image,
        modality="T2-weighted (T2w) image",
    )
    if "150 micrometer" in f.name
][0]
t2_weighted_150um_img = t2_weighted_150um.fetch()
plotting.view_img(
    t2_weighted_150um_img,
    bg_img=None,
    cmap="grey",
    symmetric_cmap=False,
    title=f"{fractional_anisotropy}\non {t2_weighted_150um}",
    threshold=0,
)

# %%
# Now, we overlay fractional anisotropy over the 150um anatomical image
plotting.view_img(
    fa_img,
    bg_img=t2_weighted_150um_img,
    cmap="magma",
    symmetric_cmap=False,
    title=f"{fractional_anisotropy}\non {t2_weighted_150um}",
    opacity=0.6,
)
