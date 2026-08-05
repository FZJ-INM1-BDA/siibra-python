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
from siibra.configuration import templateflow
from nilearn import plotting

# %%
siibra.extend_configuration(siibra.create_templateflow_configs())
for s in siibra.spaces:
    if "[TemplateFlow]" not in s.name:
        continue
    print(s.species)
    print(s.name)

# %%
MNI152NLin2009cAsym = siibra.spaces.get(
    "[TemplateFlow] ICBM 152 Nonlinear Asymmetrical template version 2009c"
)
MNI152NLin2009cAsym

# %%
print(MNI152NLin2009cAsym.name)
print(MNI152NLin2009cAsym.shortname)
print(MNI152NLin2009cAsym.publications)
print(MNI152NLin2009cAsym.description)

# %%
for template in MNI152NLin2009cAsym.volumes:
    print(template.variant)

# %%
tmp_img = MNI152NLin2009cAsym.get_template("T1w - res: 01").fetch()
plotting.view_img(tmp_img, bg_img=None, symmetric_cmap=False, cmap="gray")


# %%
# TEMP: this is just for testing. Eventually, the user will just get the map as
# usual, similar to space above
tf = templateflow.TemplateFlow()
space_spec = MNI152NLin2009cAsym.shortname.removeprefix(
    "[TemplateFlow] "
)  # TODO: remove the need for this workaround
tf.find_parcellations(space_spec)
# %%
parc = "Schaefer2018"
map_type = "labelled"
confs = tf.create_parc_and_map_configs(space_spec, parc, map_type)
for entities, conf in confs.items():
    print(entities)

# %%
mp = siibra.from_json(conf)
r = mp.regions[0]
print(r)
for idx in mp.find_indices(r):
    print(idx)
    print(mp.fetch(index=idx).shape)

# %%
plotting.view_img(
    mp.fetch(r),
    bg_img=tmp_img,
    symmetric_cmap=False,
    resampling_interpolation="nearest",
    colorbar=False,
)
