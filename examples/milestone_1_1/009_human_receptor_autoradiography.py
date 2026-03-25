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
# Autoradiography features represent 2D or 3D image data, anchored to atlases
# by image registration and queries by their bounding boxes. Since siibra can
# warp between the different human reference spaces, queries for image data work
# across template spaces. Autoradiography images are characterized by the target
# receptor type.
mni152 = siibra.spaces.get("mni152")
features_ar = siibra.features.get(mni152, "autoradiography")
siibra.features.tabulate(features_ar, ["modality"])

# %%
# Fetching the image data results in a Nifti1Image object taht can be plotted
# using standard tools such as nilearn's plotting module.
gaba_autoradiography = features_ar[0]
gaba_img = gaba_autoradiography.fetch()
plotting.plot_stat_map(
    gaba_img,
    cmap="magma",
    vmax=7500,
    draw_cross=False,
)

# %%
# siibra can use the image data to draw samples from specific regions of interest.
# We make use of this to compare GABA/BZ densities in three brain regions.
julich_brain = siibra.parcellations.get("julich 3.1")
regions = [
    julich_brain.get_region(spec)
    for spec in ["4p right", "hoc1 right", "3b right"]
]
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
for i, r in enumerate(regions):
    pmap = r.get_regional_map(mni152, "statistical")
    samples = pmap.draw_samples(10000)
    values = gaba_autoradiography.evaluate_points(samples)
    axs[i].hist(values[values > 0], bins=20, density=True)
    axs[i].set_title(r.name)
    axs[i].grid(True)
