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

title
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

desc
"""

# %%
import siibra
from ebrains_drive.client import BucketApiClient


# # %%
# siibra.retrieval.requests.EbrainsRequest.fetch_token()
# client = BucketApiClient(token=siibra.retrieval.requests.EbrainsRequest._KG_TOKEN)
# bucket = client.buckets.get_bucket("d-ed615ee5-fdaa-4f1d-8fcd-8c55d05a4e2d")


# # %%
# filepath = 'sub-XXX_ses-YYY_task-ZZZ_dir-{pa;ap}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
# filename = f"./{filepath.split('/')[-1]}"
# with open(filename, 'wb') as fp:
#     fp.write(bucket.get_file(filename))

# %%
fmri_vol = siibra.volumes.from_file(filename, time_index=[], name=filename, space='mni152')
type(fmri_vol)

# %%
julichbrain = siibra.get_map(parcellation='julich 3.1', space='mni152', maptype='statistical')
assignments = julichbrain.assign(fmri_vol, lower_threshold=0.6, split_components=False)
assignments

# %%
# source paper: https://www.nature.com/articles/s41597-021-00870-6
# 1) determine the source: https://nilab-uva.github.io/AOMIC.github.io/
# 2) select a subject
# 3) get fmri for a sub-0100_task-workingmemory_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
# 4) assign fmri to the julich brain
# 5) plot the results

# %%
# openneuro
# potentially need to warp to the mni152
url = "https://s3.amazonaws.com/openneuro.org/ds002741/sub-08/ses-mri/func/sub-08_ses-mri_task-caricatures_run-02_bold.nii.gz?versionId=mEBv6xTJbKD.n14D0fpsB7TV_MaRXmdj"
filename = 'test.nii.gz'
with open(filename, 'wb') as fp:
    fp.write(bucket.get_file(filename))
