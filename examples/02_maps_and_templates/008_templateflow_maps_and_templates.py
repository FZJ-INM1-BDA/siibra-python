# Copyright 2018-2026
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
.. _008_008_templateflow_maps_and_templates
:bdg-secondary:`Intermediate`

Extending siibra with TemplateFlow maps and templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

# %%
import siibra
from nilearn import plotting

# %%
siibra.extend_configuration(siibra.create_templateflow_configs())
siibra.spaces.dataframe

# %%
MNI152NLin6Asym = siibra.spaces.get(
    "MNI ICBM 152 non-linear 6th Generation Asymmetric"
)
MNI152NLin6Asym

# %%
print(MNI152NLin6Asym.name)
print(MNI152NLin6Asym.publications)
print(MNI152NLin6Asym.description)

# %%
for template in MNI152NLin6Asym.volumes:
    print(template.variant)

# %%
tmp_img = MNI152NLin6Asym.get_template("T1 weighted (res-01)").fetch()
plotting.view_img(tmp_img, bg_img=None, symmetric_cmap=False, cmap="gray")
