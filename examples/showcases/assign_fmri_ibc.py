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

Extrating Regionwise Signals From Activity Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

siibra integrates nilearn to allow seemles extration of signals given recordings
such as fMRI, PET, and others. This notebook downloads two fMRI images from
AOMIC-PIOP2 dataset (https://openneuro.org/datasets/ds002790/versions/2.0.0)
and compares the extraction results for different tasks.
"""

# %%
import siibra
from plotly.express import imshow
import pandas as pd

# %%
subject = "sub-0001"
dataset_base_url = "https://s3.amazonaws.com/openneuro.org/ds002790/"
folder = f"derivatives/fmriprep/{subject}/func/"
url_template = dataset_base_url + folder + "{file}"
# %%
file = f"{subject}_task-workingmemory_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
workingmemory_fmri = siibra.volumes.from_url(
    url_template.format(file=file), space="mni152", time_index=[]
)  # TODO: need to get the time index from the dataset
print(type(workingmemory_fmri))

# %%
file = f"{subject}_task-restingstate_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
restingstate_fmri = siibra.volumes.from_url(
    url_template.format(file=file), space="mni152", time_index=[]
)  # TODO: need to get the time index from the dataset


# %
workingmemory_cofounds = pd.read_csv(
    url_template.format(
        file=f"{subject}_task-workingmemory_acq-seq_desc-confounds_regressors.tsv"
    ),
    sep="\t",
)
restingstate_cofounds = pd.read_csv(
    url_template.format(
        file=f"{subject}_task-restingstate_acq-seq_desc-confounds_regressors.tsv"
    ),
    sep="\t",
)

# %%
# We now select the map we would like to extract the signals from
julichbrain = siibra.get_map("julich 3.1", "mni152")

# %%
# `extract_signals` method takes a 3D/4D volume and returns the results as a
# `pandas.DataFrame` where the columns are the regions extracted and the rows
# represnt 4th dimension of the input volume.
# (TODO: consider allowing resampling target to be the map instead of the input.
# Nilearn handles this if kwarg allowed. One could potentiall allow nilearn.Masker
# kwargs as makser_kwargs and docstring could lead to relevant nilearn page.)
workingmemory_signals = julichbrain.extract_signals_with_nilearn(
    workingmemory_fmri,
    confounds=workingmemory_cofounds["csf"].values,
)
workingmemory_signals

# %%
# we can simply visualize the results as a carpet plot
imshow(workingmemory_signals.T, origin="lower")

# %%
# We can make use of pandas built-in functions to get a summary of the results
workingmemory_signals_stats = workingmemory_signals.describe().T.sort_values(
    "mean", ascending=False
)
workingmemory_signals_stats

# %%
# Similarly, we extract the signals for the rest state
restingstate_signals = julichbrain.extract_signals_with_nilearn(
    restingstate_fmri,
    confounds=restingstate_cofounds["csf"].values,
)
restingstate_signals_stats = restingstate_signals.describe().T.sort_values(
    "mean", ascending=False
)
restingstate_signals_stats

# %%
# Using plotting backend of pandas, we can plot the data to compare the first 10
restingstate_signals_stats.iloc[:10, :].plot(
    title="resting state", kind="barh", y="mean", xerr="std"
)
workingmemory_signals_stats.iloc[:10, :].plot(
    title="working memory", kind="barh", y="mean", xerr="std"
)

# %%
# As an alteranative, pmaps can be used to extract signals
difumo64_pmaps = siibra.get_map("difumo 64", "mni152", "statistical")
difumo64_pmaps.extract_signals_with_nilearn(workingmemory_fmri)


# %%
# TODO: Find an fMRI for assignment where there are clusters and fluctuations
# julichbrain = siibra.get_map(parcellation='julich 3.1', space='mni152', maptype='statistical')
# assignments = julichbrain.assign(fmri_vol, lower_threshold=0.6, split_components=False)
# assignments


# sub-0100_task-workingmemory_acq-seq_space-fsaverage5_hemi-L.func.gii
#     ├── sub-0100_task-workingmemory_acq-seq_space-fsaverage5_hemi-R.func.gii
