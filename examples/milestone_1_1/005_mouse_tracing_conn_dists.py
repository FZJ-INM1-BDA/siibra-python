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
Mouse tracing connectivity distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# Tracing connectivity features are anchored to brain regions matching their 
# injection sites.
# A query by the Allen mouse brain atlas yields many features
# stemming from three different datasets.
amba_v3 = siibra.parcellations.get("Allen Mouse v3 2017")
features_tracing = siibra.features.get(
    amba_v3, "tracing connectivity distribution"
)
print(f"Found {len(features_tracing)} tracing connectivity distribution features.")

# %%
# Compile an overview of the retrieved features in terms of their
# anchored brain region, subject specification and origin dataset.
feature_table = siibra.features.compile_feature_table(
    features_tracing, ["anchor", "subject", "datasets"],
    converters={"anchor": str, "datasets": lambda ds: next(iter(ds)).name},
)
feature_table

# %%
# The table reveals that features originate from 18 different brain areas.
feature_table.region.drop_duplicates().to_frame().reset_index(drop=True)

# %%
# Furthermore, features originate from three different datasets, including one for
# wild type and another for Cre-transgenic mice.
feature_table.dataset.drop_duplicates().to_frame().reset_index(drop=True)

# %%
# The subject specification is a combination of subject id and subcortical projection target.
# We split the field in the table.
feature_table['target'] = [s.split('_')[-1] for s in feature_table.subject]
feature_table['subject'] = ['_'.join(s.split('_')[:-1]) for s in feature_table.subject]

# In total, fourteen different subcortical projection targets are represented.
feature_table.target.drop_duplicates().to_frame().reset_index(drop=True)

# %%
# The feature query can be refined to tracing connectivity from a specific brain area.
region_acd = amba_v3.get_region("Anterior cingulate area, dorsal part")
features_tracing_acd = siibra.features.get(
    region_acd, "tracing connectivity distribution"
)

# %%
# Tracing results could then be compared between wild type
# and Cre-transgenic mice for different subcortical targets:
subcortical_regions = feature_table.target.unique()[:4]
allen_mouse_template = siibra.get_template("mouse").fetch()
fig, axs = plt.subplots(len(subcortical_regions), 1, figsize=(19, 24))
for i, target in enumerate(subcortical_regions):
    selection = [f for f in features_tracing_acd if target in f.subject]
    tcd_tg = [f for f in selection if "transgenic" in f.description][0]
    tcd_wt = [f for f in selection if "wild-type" in f.description][0]
    display = plotting.plot_img(
        img=allen_mouse_template,
        bg_img=None,
        cmap="gray",
        title=target,
        cut_coords=tcd_wt.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
        colorbar=False,
    )
    display.add_markers(
        tcd_wt.data,
        marker_color="r",
        marker_size=1,
        label="Wild type"
    )
    display.add_markers(
        tcd_tg.data,
        marker_color="b",
        marker_size=1,
        label="Cre-transgenic",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(1.5, 0.6), fontsize="x-large")

# %%
