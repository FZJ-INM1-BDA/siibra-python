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
Human receptor autoradiography
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
import numpy as np
import nibabel
from nilearn import plotting

# %%
mni152 = siibra.spaces["mni152"]
gaba_autoradiography = siibra.features.get(
    mni152, siibra.features.molecular.AutoradiographyVolumeOfInterest
)[0]
print(gaba_autoradiography.modality)

plotting.view_img(
    gaba_autoradiography.fetch(),
    cmap="magma",
    symmetric_cmap=False,
)

# %%
gaba_peaks = gaba_autoradiography.find_peaks()
print(f"Found {len(gaba_peaks)} peaks")
peak_values = gaba_autoradiography.evaluate_points(gaba_peaks)
maximum_peak = gaba_peaks[np.argmax(peak_values)]
print(f"Global maximum: {maximum_peak}")

# %%
julich_pmaps = siibra.get_map("julich 3.1", mni152, "statistical")
assignments = julich_pmaps.assign(maximum_peak)
assignments

# %%
fo1_R = assignments.loc[0, "region"]
fo1_R_mask = fo1_R.get_regional_mask(mni152)

# %%
proton_density = siibra.features.get(
    mni152, siibra.features.generic.Image, modality="Proton density (PD) image"
)[0]
pd_img = proton_density.fetch()
plotting.plot_img(pd_img, cmap="magma")

# %%
intersection_mask = proton_density.intersection(fo1_R_mask).fetch()
pd_img_masked_arr = np.asanyarray(pd_img.dataobj)
pd_img_masked_arr[~np.asanyarray(intersection_mask.dataobj, dtype=bool)] = 0
pd_img_masked = nibabel.Nifti1Image(pd_img_masked_arr, pd_img.affine)
cut_coords = fo1_R.compute_centroids(mni152)[0].coordinate
plotting.view_img(
    pd_img_masked,
    symmetric_cmap=False,
    cut_coords=cut_coords,
    cmap="magma",
)

# %%
voi = fo1_R_mask.get_boundingbox(clip=True)
pd_img_hi_res_voi = proton_density.fetch(voi=voi)
plotting.view_img(
    pd_img_hi_res_voi,
    bg_img=None,
    black_bg=True,
    symmetric_cmap=False,
    cmap="magma",
    cut_coords=cut_coords,
)
