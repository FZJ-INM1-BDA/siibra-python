# Copyright 2018-2023
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
Adding a custom parcellation map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you might want to use a custom parcellation map to perform feature queries in siibra.
This brings some challenges as we will see. For this example, we retrieve a
freely available von Economo map in MNI space from Martijn van den Heuvel's lab.
"""


# %%
import siibra
import re  # we need this for text file parsing below
from nilearn import plotting

# %%
# Load the custom parcellation map from the online resource. For the retrieval,
# siibra's zipfile connector is very helpful. We can use the resulting NIfTI
# to create a custom parcellation map inside siibra.

# connect to the online zip file
conn = siibra.retrieval.ZipfileConnector(
    "http://www.dutchconnectomelab.nl/economo/economoMNI152volume.zip"
)

# the NIfTI file is easily retrieved:
nifti_map = conn.get("economoMNI152volume/economoMNI152volume.nii.gz")

# The text file with label mappings has a custom format.
# We provide a simple decoder to extract the list of region/label pairs.
decoder = lambda b: [
    re.split(r'\s+', line)[:2]       # fields are separated by one or more whitespaces
    for line in b.decode().split('\n')
    if re.match(r'^\d\d\d\d', line)  # only lines starting with 4-digit labels are relevant
]
labels = conn.get("economoMNI152volume/economoLUT.txt", decode_func=decoder)

# Now we use this to add a custom map to siibra.
# Note that this assumes our external knowledge that the map is in MNI152 space.
custom_map = siibra.add_nifti_map(
    name="Von Economo Atlas",
    nifti_map=nifti_map,
    space_spec='mni152',
    regionnames=[name.replace('ctx-rh-', 'right ').replace('ctx-lh-', 'left ') for _, name in labels],
    regionlabels=[int(label) for label, _ in labels]
)

# let's plot the final map
plotting.plot_roi(custom_map.fetch())


# %%
# We can already use this map to find spatial features, such as BigBrain
# intensity profiles.
custom_region = custom_map.parcellation.get_region('fcbm')
profiles = siibra.features.get(
    custom_region,
    siibra.features.cellular.BigBrainIntensityProfile
)
print(f"{len(profiles)} intensity profiles found.")
