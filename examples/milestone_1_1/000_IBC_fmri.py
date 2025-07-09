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
voneconomo = siibra.parcellations["von economo"]
r = voneconomo.get_region("FC: Area frontalis intermedia left")
functional_fingerprints = siibra.features.get(
    r, siibra.features.functional.FunctionalFingerprint
)
for f in functional_fingerprints:
    print(f.anchor)
fe_left_fp = functional_fingerprints[0]

# %%
fe_left_fp.data

# %%
fe_left_fp.plot(backend="plotly")

# %%
functional_fingerprints = siibra.features.get(
    voneconomo, siibra.features.functional.FunctionalFingerprint
)
functional_fingerprints_ve = pd.concat((f.data for f in functional_fingerprints), axis=1)
functional_fingerprints_ve

# %%
selected_task_and_label = ("ArchiSocial", "triangle_mental-random")
voneconomo_map = voneconomo.get_map("MNI 152")
colored_map = voneconomo_map.colorize(
    functional_fingerprints_ve.loc[selected_task_and_label].to_dict()
).fetch()
plotting.view_img(
    colored_map,
    symmetric_cmap=False,
    cmap="magma",
    resampling_interpolation="nearest",
)
