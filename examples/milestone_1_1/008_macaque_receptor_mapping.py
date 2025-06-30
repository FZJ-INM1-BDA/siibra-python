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
Macaque receptor mapping
~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
parc = siibra.parcellations["mebrains"]
print(parc.name)

# %%
receptor_density = siibra.features.get(parc, siibra.features.generic.Tabular)[0]
print(receptor_density.modality)

# %%
receptor_density.data

# %%
receptor = "AMPA"
receptor_density.plot(y=receptor, backend="plotly")

# %%
mp = parc.get_map("mebrains")
region_density_lookup = {}
for r in receptor_density.data.index:
    try:
        region_density_lookup[
            parc.get_region(r.replace("area_", "")).name + " left"
        ] = receptor_density.data[receptor].loc[r]
        region_density_lookup[
            parc.get_region(r.replace("area_", "")).name + " right"
        ] = receptor_density.data[receptor].loc[r]
    except ValueError as e:
        print(e)
receptor_denisty_map = mp.colorize(region_density_lookup)
plotting.view_img(
    receptor_denisty_map.fetch(),
    bg_img=mp.get_resampled_template().fetch(),
    cmap="magma",
    symmetric_cmap=False,
)
