# Copyright 2018-2021
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
Parcellation-based functional data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides access to parcellation-averaged functional data such as
blood-oxygen-level-dependent (BOLD) signals.
"""

# %%
import siibra
# sphinx_gallery_thumbnail_number = 1

# %%
# We start by selecting an atlas parcellation.
julich_brain = siibra.parcellations.get("julich 2.9")

# %%
# The tables are queried as expected, using `siibra.features.get`, passing
# the parcellation as a concept. Here, we query for regional BOLD signals.
# Since a single query may yield hundreds of signal tables for different
# subjects of a cohort and paradigms, siibra groups them as elements into
# :ref:`CompoundFeatures<compoundfeatures>`. Let us select "rfMRI_REST1_LR_BOLD"
# paradigm.
features = siibra.features.get(julich_brain, siibra.features.functional.RegionalBOLD)
for f in features:
    print(f.name)
    if f.paradigm == "rfMRI_REST1_LR_BOLD":
        cf = f
        print(f"Selected: {cf.name}'\n'" + cf.description)

# %%
# We can select a specific element by integer index
print(cf[0].name)
print(cf[0].subject)  # Subjects are encoded via anonymized ids


# %%
# The signal data is provided as pandas DataFrames with region objects as
# columns and indices as as a timeseries.
table = cf[0].data
table[julich_brain.get_region("hOc3v left")]  # list the data for 'hOc3v left'

# %%
# We can visualize the signal strength per region by time via a carpet plot.
# In fact, `plot_carpet` method can take a list of regions to display the
# data for selected regions only.
selected_regions = [
    'SF (Amygdala) left', 'SF (Amygdala) right', 'Area Ph2 (PhG) left',
    'Area Ph2 (PhG) right', 'Area Fo4 (OFC) left', 'Area Fo4 (OFC) right',
    'Area 7A (SPL) left', 'Area 7A (SPL) right', 'CA1 (Hippocampus) left',
    'CA1 (Hippocampus) right', 'CA1 (Hippocampus) left', 'CA1 (Hippocampus) right'
]
cf[0].plot_carpet(regions=selected_regions)
# %%
# Alternatively, we can visualize the mean signal strength per region:
cf[0].plot(regions=selected_regions, backend='plotly')
