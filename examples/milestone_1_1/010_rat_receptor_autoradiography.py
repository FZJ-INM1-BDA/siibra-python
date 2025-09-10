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
Rat receptor autoradiogarphy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# Receptor autoradiography sections are anchored in to bounding boxes in
# reference spaces. To see all availabe options for rat, query for with waxholm
# space. Then, display dataset information:
sections_at_HPWM = siibra.features.get(siibra.spaces["waxholm"], "autoradiography")
datasets = {
    "".join(ds.name for ds in tcd.datasets): "".join(tcd.urls)
    for tcd in sections_at_HPWM
}
for name, doi in datasets.items():
    print("Name:", name)
    print("DOI:", doi)

# %%
# The names of the datasets reveal that they are divided by neurotransmitters.
# For a detailed look at the serotonin, we query for sections intersecting with
# the region "Hippocampal white matter" (TODO: check for citations).
# `siibra` will automatically merge the maps of the children of hippocampal
# white matter create a mask to compare against the sections.
sections_at_HPWM = siibra.features.get(
    siibra.get_region("waxholm", "Hippocampal white matter"), "autoradiography"
)
print("Sections found:", len(sections_at_HPWM))
serotonin_sections_at_HPWM = [s for s in sections_at_HPWM if "serotonin" in s.name]
print("Sections from the serotonin study:", len(serotonin_sections_at_HPWM))

# %%
# Next, display them using matplotlib:
fig = plt.figure(figsize=(15, 36))
for i, section in enumerate(serotonin_sections_at_HPWM):
    img = section.fetch()
    plt.subplot(9, 3, i + 1)
    plt.imshow(img.dataobj.squeeze().T, aspect="equal")
    plt.axis("off")
