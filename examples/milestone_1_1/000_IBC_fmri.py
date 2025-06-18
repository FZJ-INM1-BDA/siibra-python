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

# %%
p = siibra.parcellations["von economo"]
functional_fingerprints = siibra.features.get(
    p, siibra.features.functional.FunctionalFingerprint
)[0]


# %%
f = functional_fingerprints.get_element(
    p.get_region("FC: Area frontalis intermedia left")
)
f.data

# %%
f.plot(backend="plotly")

# %%
functional_fingerprints.data

# %%
selected_task_and_label = ("ArchiSocial", "triangle_mental-random")
mp = p.get_map("MNI 152")
colored_map = mp.colorize(
    functional_fingerprints.data.loc[selected_task_and_label].to_dict()
).fetch()
plotting.view_img(
    colored_map,
    symmetric_cmap=False,
    cmap="magma",
    resampling_interpolation="nearest",
)
