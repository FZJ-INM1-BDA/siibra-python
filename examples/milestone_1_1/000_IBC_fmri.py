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
IBC - fMRI Data
~~~~~~~~~~~~~~~
"""
# %%
import siibra
from nilearn import plotting
import pandas as pd

# %%
julichbrain = siibra.parcellations["julich 3.1"]
r = julichbrain.get_region("area 3b left")
functional_fingerprints = siibra.features.get(
    r, siibra.features.functional.FunctionalFingerprint
)
for f in functional_fingerprints:
    print(f.anchor)
area3b_left_fp = functional_fingerprints[0]

# %%
area3b_left_fp.data

# %%
area3b_left_fp.plot(backend="plotly")

# %%
functional_fingerprints = siibra.features.get(
    julichbrain, siibra.features.functional.FunctionalFingerprint
)
functional_fingerprints_ve = pd.concat((f.data for f in functional_fingerprints), axis=1)
functional_fingerprints_ve

# %%
selected_task_and_label = ("ArchiSocial", "triangle_mental-random")
julichbrain_mni152_map = julichbrain.get_map("MNI 152")
colored_map = julichbrain_mni152_map.colorize(
    functional_fingerprints_ve.loc[selected_task_and_label].to_dict()
).fetch()
plotting.view_img(
    colored_map,
    symmetric_cmap=False,
    cmap="magma",
    resampling_interpolation="nearest",
)
