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
1000 Brains connectivity
~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra

# %%
p = siibra.parcellations.get("julich 3.1")
features = siibra.features.get(p, "streamlinecounts")
for f in features:
    print(f.cohort)

# %%
cf = list(filter(lambda f: f.cohort == "1000BRAINS", features))[0]
print(cf.name)

# %%
for f in cf:
    print(f.subject)

# %%
cf[0].data

# %%
cf[0].plot(regions=cf[0].regions[0:50], reorder="average")

# %%
cf[0].plot(regions="3b left", backend="plotly", min_connectivity=400)
