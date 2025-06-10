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
"""

# %%
import siibra
from nilearn import plotting

# %%
space = siibra.spaces.get("mouse")

# %%
fts = siibra.features.get(space, "PointDistribution")
fts[0][0].data

# %%
display = plotting.plot_img(
    img=siibra.get_template("mouse").fetch(resolution_mm=-1),
    bg_img=None,
    cmap="gray",
    title=fts[0][0].subject,
)
display.add_markers(fts[0][0].data.values)
