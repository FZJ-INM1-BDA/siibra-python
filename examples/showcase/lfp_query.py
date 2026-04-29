# Copyright 2018-2023
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
Showcase: High-resolution Rat Local Field Potential Atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WIP
"""

# %%
import siibra

# %%
waxholm = siibra.parcellations.get("waxholm")
caudate_putamen = waxholm.get_region("Caudate putamen")

# %%
lfp_caudate_putamen = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotential
)
print(f"Found {len(lfp_caudate_putamen)} local field potentials")

# %%
specs = {
    "pharmacology": "baseline",
    "signal_quality": "typical"
}
fts_w_specs = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotential,
    **specs
)
print(f"Found {len(fts_w_specs)} local field potentials with specs: {specs}")

# %%
specs = {
    "pharmacology": "baseline",
    "signal_quality": "atypical"
}
fts_w_specs = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotential,
    **specs
)
print(f"Found {len(fts_w_specs)} local field potentials with specs: {specs}")

# %%
f = lfp_caudate_putamen[0]
print("subject:", f.subject)
print("session:", f.session)
print("pharmacology:", f.pharmacology)
print("pathology:", f.pathology)
print("signal_quality:", f.signal_quality)

# %%
lfp_spectrum_caudate_putamen = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotentialSpectrum,
    pathology=None,
    pharmacology="baseline",
    signal_quality="typical"
)
lfp_spectrum_caudate_putamen
# %%
# lfp_spectrum_caudate_putamen[0].plot()
