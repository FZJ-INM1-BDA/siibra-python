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
# Autoradiography features contain image data, typically volumes or 2D sections.
# Here, we query for autoradiography features registered to MNI 152 2009c
# Asym Nonl space. Autoradiography images are characterized by a receptor. Here,
# we select the first volume of Gaba/BZ distribution:
mni152 = siibra.spaces["mni152"]
autoradiography_images = siibra.features.get(mni152, "autoradiography")
for f in autoradiography_images:
    print(f.modality)
    if "GABA/BZ" in f.modality:
        gaba_autoradiography = f
        break

# %%
# Image data can be fetched the online resource and can be plotted on top of the
# MNI 152 2009c Nonl Asym template. Plot reveals the image covers the complete
# right hemisphere.
img = gaba_autoradiography.fetch()
plotting.plot_stat_map(
    img,
    cmap="magma",
    title=gaba_autoradiography.name,
    threshold=0,
    vmin=0,
    vmax=7500,
)

# %%
# siibra can use the image data to draw samples from spefic regions of interest.
# Here, we compare GABA/BZ densities in 4p right, hoc1 right, and 3b right.
julich_brain = siibra.parcellations["julich 3.1"]
regions = [
    julich_brain.get_region(spec)
    for spec in ["4p right", "hoc1 right", "3b right"]
]
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
for i, r in enumerate(regions):
    pmap = r.get_regional_map(mni152, "statistical")
    samples = pmap.draw_samples(10000)
    values = gaba_autoradiography.evaluate_points(samples)
    axs[i].hist(values[values > 0], bins=50, density=True)
    axs[i].set_title(r.name)
