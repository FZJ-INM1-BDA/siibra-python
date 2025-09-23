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
from nilearn import plotting, image
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/gaba_autoradiography.png'

# %%
# Begin by querying for autoradiography volumes registered to MNI 152 2009c
# Asym Nonl space and select the volume of Gaba/BZ distribution:
mni152 = siibra.spaces["mni152"]
autoradiography_images = siibra.features.get(
    mni152, siibra.features.molecular.AutoradiographyVolumeOfInterest
)
for f in autoradiography_images:
    print(f.name)
    if "GABA/BZ" in f.name:
        gaba_autoradiography = f

# %%
# We can then plot it over the template, however, the volume has a higher
# resolution. So we fetch in 600um resolution and upsample the template from 1mm
# to display it
img = gaba_autoradiography.fetch(resolution_mm=0.6)
upsampled_template_img = image.resample_to_img(mni152.get_template().fetch(), img)
plotting.view_img(
    img,
    bg_img=upsampled_template_img,
    cmap="magma",
    symmetric_cmap=False,
    vmin=1e-7,
    vmax=7500,
)

# %%
# We can make use of siibra to directly draw samples from selected parts of the
# volume. For this example, draw samples from boxes surrounding regions
# "4p right", "hoc1 right", and "3b right". Then, compare the distributions
# using matplotlib:
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
    sample_values = gaba_autoradiography.evaluate_points(samples, max_bytes=1 * 1024**3)
    axs[i].hist(sample_values, bins=100)
    axs[i].set_title(r.name)
    axs[i].set_xlim((1e-7, 7500))
