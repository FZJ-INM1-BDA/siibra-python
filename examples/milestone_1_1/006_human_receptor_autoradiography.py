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
Human receptor autoradiography
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/gaba_autoradiography.png'

# %%
mni152 = siibra.spaces["mni152"]
autoradiography_images = siibra.features.get(
    mni152, siibra.features.molecular.AutoradiographyVolumeOfInterest
)
for f in autoradiography_images:
    print(f.name)

# %%
gaba_autoradiography = autoradiography_images[0]
print(gaba_autoradiography.modality)

plotting.view_img(
    gaba_autoradiography.fetch(),
    cmap="magma",
    symmetric_cmap=False,
)

# %%
julich_brain = siibra.parcellations["julich 3.1"]
regions = [
    julich_brain.get_region(spec) for spec in ["4p right", "hoc1 right", "3b right"]
]
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
for i, r in enumerate(regions):
    samples = gaba_autoradiography.draw_samples(
        25000,
        voi=r.get_boundingbox(mni152),
        max_bytes=1 * 1024**3,
    )
    axs[i].hist(gaba_autoradiography.evaluate_points(samples, max_bytes=1 * 1024**3))
    axs[i].set_title(r.name)
    axs[i].set_xlim((0, 7500))
