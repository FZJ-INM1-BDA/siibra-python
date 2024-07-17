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

from typing import Iterable
from urllib.parse import quote

from .git_fetcher import GitHttpRepository
from .tar_fetcher import TarRepository


class GitlabRepository(GitHttpRepository):

    PER_PAGE = 100

    def __init__(
        self, gitlab_server: str, projectpath: str, reftag: str, eager=False
    ) -> None:
        if gitlab_server.startswith("https://"):
            gitlab_server = gitlab_server.replace("https://", "")
        url = f"https://{gitlab_server}/{quote(projectpath, safe='')}.git"
        super().__init__(url, reftag)
        self.gitlab_server = gitlab_server
        self.projectpath = projectpath
        self.reftag = reftag
        self.tar_repo = TarRepository(
            f"{self.gitlabapi_url}/archive.tar.gz?sha={reftag}", gzip=True
        )
        if eager:
            self.warmup()

    @property
    def gitlabapi_url(self):
        return f"https://{self.gitlab_server}/api/v4/projects/{quote(self.projectpath, safe='')}/repository"

    def search_files(self, prefix: str = None) -> Iterable[str]:
        if self.is_warm:
            yield from self.tar_repo.search_files(prefix)
        page = 1
        while True:
            resp = self.sess.get(
                f"{self.gitlabapi_url}/tree?ref={self.reftag}?per_pagae={self.PER_PAGE}&page={page}&recursive=True"
            )
            resp.raise_for_status()
            page += 1
            results = [
                obj["path"]
                for obj in resp.json()
                if obj["type"] == "blob" and obj["name"].startswith(prefix or "")
            ]
            yield from results
            if len(results) < self.PER_PAGE:
                break

    @property
    def unpacked_dir(self):
        return self.tar_repo.unpacked_dir

    def warmup(self, *args, **kwargs):
        self.tar_repo.warmup()
