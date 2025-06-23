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
Mice point dist
~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
space = siibra.spaces.get("mouse")

# %%
cf = siibra.features.get(space, "PointDistribution")[0]
print(cf.modality)
print(cf.tracer)
cf[0].data


# %%
for f in cf:
    print(f.subject)

# %%
display = plotting.plot_img(
    img=siibra.get_template("mouse").fetch(resolution_mm=-1),
    bg_img=None,
    cmap="gray",
    title=f"red: {cf[0].subject}, blue: {cf[-1].subject}",
    cut_coords=cf[0].data.values.mean(axis=0)
)
display.add_markers(cf[0].data.values, marker_color="r")
display.add_markers(cf[-1].data.values, marker_color="b")
