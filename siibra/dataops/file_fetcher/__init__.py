# Copyright 2018-2024
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

from .tar_fetcher import TarRepository, TarDataOp
from .git_fetcher import GitHttpRepository
from .zip_fetcher import ZipRepository, ZipDataOp
from .github_fetcher import GithubRepository
from .gitlab_fetcher import GitlabRepository
from .local_directory_fetcher import LocalDirectoryRepository, RemoteLocalDataOp
