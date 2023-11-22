# Copyright 2018-2021
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
Compound features
~~~~~~~~~~~~~~~~~

Some features such as connectivity matrices have attributes siibra can use to
combine them into one set, call CompoundFeatures. They allow easy handling of
similar features.
"""

# %%
# Query mechanism for CompoundFeatures is the same as any other. Only difference
# is that the resulting Feature object has a few extra functionality. Let us
# demonstrate it with connectivity matrices. They also inherit joint attributes
# (with the same value) from their elements. Using these, we can distinguish
# different CompoundFeatures of the same feature type.
import siibra
features = siibra.features.get(siibra.parcellations["julich 2.9"], "StreamlineLengths")
for f in features:
    print("Compounded feature type:", f.feature_type)
    print(f.name)
    if f.cohort == "1000BRAINS":
        cf = f
        print(f"Selected: {cf.name}")

# %%
# Each of these CompoundFeatures have StreamlineLengths features as elements.
# We can access to these elements via an integer index or by their key unique
# to a CompoundFeature using `get_element`.
print(cf[5].name)
print(cf.get_element('0031_2').name)

# %%
# The element key changes based on the type of features that make up a
# CompoundFeature. CompoundFeatures composed of StreamlineLengths obtain their
# element key from the subject id.
for i, f in enumerate(cf[:10]):  # we can iterate over elements of a CompoundFeature
    print(f"Element index: {cf.indices[i]}, Subject: {f.subject}")

# %%
# Meanwhile, receptor density profiles employ receptor names.
cf = siibra.features.get(siibra.get_region('julich', 'hoc1'), 'receptor density profile')[0]
for i, f in enumerate(cf):
    print(f"Element index: {cf.indices[i]}, receptor: {f.receptor}")

# %%
# So to get the receptor profile on HOC1 for GABAB
cf.get_element("GABAB").data

# %%
# Similarly, to plot
cf.get_element("GABAB").plot()
