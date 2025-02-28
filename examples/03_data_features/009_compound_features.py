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
Compound features
~~~~~~~~~~~~~~~~~

Some features such as connectivity matrices have attributes siibra can use to
combine them into one feature object, called `CompoundFeature`. Compound features
contain all the features making them up as elements and allow easy access to each
element.
"""

# %%
# Compound features naturally result from a feature query for certain feature types.
# For example, connectivty matrices usually provided for each subject, however,
# having them as seperate featuers make it difficult to work with them. But as a
# compound feature, they inherit the joint attributes from their elements. But
# siibra will not compound different cohorts for example. Let us demonstrate:
import siibra
features = siibra.features.get(siibra.parcellations["julich 2.9"], "StreamlineLengths")
for f in features:
    print("Compounded feature type:", f.feature_type)
    print(f.name)
    # let us select the 1000 Brains cohort
    if f.cohort == "1000BRAINS":
        cf = f
        print(f"Selected: {cf.name}")

# %%
# Each of these features consist of streamline lengths features corresponding to
# different subjects. An element can be selected via an integer index or by
# their index to a CompoundFeature using `get_element`:
print(cf[5].name)
print(cf.get_element('0031_2').name)

# %%
# The indicies of this compound feature corresponds to the the subject ids:
for i, f in enumerate(cf[:10]):  # we can iterate over elements of a CompoundFeature
    print(f"Element index: {cf.indices[i]}, Subject: {f.subject}")

# %%
# We can also obtain the averaged data (depends on the underlying feature type) by
# as you would normally access the data of a feature
cf.data

# %%
# Meanwhile, receptor density profiles employ receptor names as indices.
cf = siibra.features.get(siibra.get_region('julich', 'hoc1'), 'receptor density profile')[0]
for i, f in enumerate(cf):
    print(f"Element index: {cf.indices[i]}, receptor: {f.receptor}")

# %%
# So to get the receptor profile on HOC1 for GABAB we can do
cf.get_element("GABAB").data

# %%
# Similarly, to plot
cf.get_element("GABAB").plot()
