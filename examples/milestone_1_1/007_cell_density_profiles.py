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
Cell density Profiles
~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra

# %%
profiles = siibra.features.get(
    siibra.get_region("julich 3.1", "spl 7m"), "cell density profile"
)
for pf in profiles:
    print(pf.name)


# %%
pf.data


# %%
pf.plot(y="cell_size_mean_um2", error_y="cell_size_std_um2", backend="plotly")
