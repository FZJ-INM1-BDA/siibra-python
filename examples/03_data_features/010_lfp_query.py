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
High-resolution Rat Local Field Potential Atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local field potentials (LFPs) capture electrophysiological activity recorded
from neural tissue. In this example, we query LFP recordings and spectra from
a high-resolution rat atlas dataset linked to regions of the Waxholm Space
atlas.
"""

# %%
import siibra

# %%
# We start by selecting the Waxholm Space parcellation and choosing the
# caudate putamen as the anatomical region of interest.
waxholm = siibra.parcellations.get("waxholm")
caudate_putamen = waxholm.get_region("Caudate putamen")

# %%
# Query all local field potential features linked to the caudate putamen.
lfp_caudate_putamen = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotential
)
print(f"Found {len(lfp_caudate_putamen)} local field potentials")

# %%
# LFP features provide metadata fields that can be used to filter the results.
# Here, we restrict the query to baseline recordings with typical signal quality.
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
# The same metadata options can also be used to search for atypical recordings.
# This allows comparing different subsets of the available electrophysiological
# data.
specs["signal_quality"] = "atypical"
fts_w_specs = siibra.features.get(
    caudate_putamen,
    siibra.features.functional.LocalFieldPotential,
    **specs
)
print(f"Found {len(fts_w_specs)} local field potentials with specs: {specs}")

# %%
# Inspect the metadata of one LFP feature. These fields describe the subject,
# recording session, pharmacological condition, pathology, and signal quality.
f = lfp_caudate_putamen[0]
print("subject:", f.subject)
print("session:", f.session)
print("pharmacology:", f.pharmacology)
print("pathology:", f.pathology)
print("signal_quality:", f.signal_quality)


# %%
# We can plot spectra using any regions including the whole Waxholm parcellation
# using `plot_spectrum`` method. Here selecting atypical recordings from the
# lesioned hemisphere in 6-OHDA hemilesioned animals.
specs = dict(
    pathology="lesioned hemisphere in 6-OHDA hemilesioned animal",
    pharmacology="baseline",
    signal_quality="atypical",
)
lfp_spectrum_w_specs = siibra.features.get(
    waxholm,
    siibra.features.functional.LocalFieldPotential,
    **specs
)
siibra.features.functional.LocalFieldPotential.plot_spectrum(
    lfp_spectrum_w_specs,
    spectrum_type="spectrogram_rhythmic"
)
