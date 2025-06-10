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
sulci atlas
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
p = siibra.parcellations.get("Sulci atlas")
p.render_tree()

# %%
for s in siibra.maps.dataframe.query(f'parcellation == "{p.name}"')["space"]:
    mp = p.get_map(s)
    plotting.plot_img(
        img=mp.fetch(resolution_mm=1),
        bg_img=mp.space.get_template().fetch(resolution_mm=1),
        cmap=mp.get_colormap(),
        title=mp.name,
    )
plt.show()
