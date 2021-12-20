# Copyright 2018-2021
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

from .requests import DECODERS, HttpRequest
from .. import logger
from abc import ABC, abstractmethod
from urllib.parse import quote
import base64
from tqdm import tqdm
import json


class RepositoryConnector(ABC):
    """
    Base class for repository connectors.
    """

    def __init__(self, base_url):
        self.base_url = base_url

    @abstractmethod
    def search_files(folder: str, suffix: str, recursive: bool = False):
        pass

    @abstractmethod
    def _build_url(self, folder: str, filename: str):
        pass

    def _decode_response(self, response, filename):
        # see if we find a default encoder
        suitable_decoders = [
            dec for sfx, dec in DECODERS.items() if filename.endswith(sfx)
        ]
        if len(suitable_decoders) > 0:
            assert len(suitable_decoders) == 1
            return suitable_decoders[0](response)
        else:
            return response

    def get(self, filename, folder="", decode_func=None):
        """ Get a file right away. """
        return self.get_loader(filename, folder, decode_func).data

    def get_loader(self, filename, folder="", decode_func=None):
        """ Get a lazy loader for a file, for executing the query
        only once loader.data is accessed. """
        if decode_func is None:
            return HttpRequest(
                self._build_url(folder, filename),
                lambda b: self._decode_response(b, filename),
            )
        else:
            return HttpRequest(self._build_url(folder, filename), decode_func)

    def get_loaders(
        self, folder="", suffix=None, progress=None, recursive=False, decode_func=None
    ):
        """
        Returns an iterator with lazy loaders for the files in a given folder.
        In each iteration, a tuple (filename,file content) is returned.
        """
        fnames = self.search_files(folder, suffix, recursive)
        result = [
            (fname, self.get_loader(fname, decode_func=decode_func)) for fname in fnames
        ]
        all_cached = all(_[1].cached for _ in result)
        if progress is None or all_cached:
            return result
        else:
            return tqdm(result, total=len(fnames), desc=progress, disable=logger.level>20)


class GitlabConnector(RepositoryConnector):
    def __init__(self, server: str, project: int, reftag: str, skip_branchtest=False):
        # TODO: the query builder needs to check wether the reftag is a branch, and then not cache.
        assert server.startswith("http")
        RepositoryConnector.__init__(
            self, base_url=f"{server}/api/v4/projects/{project}/repository"
        )
        self.reftag = reftag
        self._per_page = 100
        self._branchloader = HttpRequest(
            f"{self.base_url}/branches", DECODERS[".json"], refresh=True
        )
        self._tag_checked = True if skip_branchtest else False
        self._want_commit_cached = None

    def __str__(self):
        return f"{self.__class__.__name__} {self.base_url} {self.reftag}"

    @property
    def want_commit(self):
        if not self._tag_checked:
            try:
                matched_branches = list(
                    filter(lambda b: b["name"] == self.reftag, self.branches)
                )
                if len(matched_branches) > 0:
                    self._want_commit_cached = matched_branches[0]["commit"]
                    logger.debug(
                        f"{self.reftag} is a branch of {self.base_url}! Want last commit "
                        f"{self._want_commit_cached['short_id']} from "
                        f"{self._want_commit_cached['created_at']}"
                    )
                self._tag_checked = True
            except Exception as e:
                print(str(e))
                print("Could not connect to gitlab server!")
        return self._want_commit_cached

    @property
    def branches(self):
        return self._branchloader.data

    def _build_url(self, folder="", filename=None, recursive=False, page=1):
        ref = self.reftag if self.want_commit is None else self.want_commit["short_id"]
        if filename is None:
            pathstr = "" if len(folder) == 0 else f"&path={quote(folder,safe='')}"
            return f"{self.base_url}/tree?ref={ref}{pathstr}&per_page={self._per_page}&page={page}&recursive={recursive}"
        else:
            pathstr = filename if folder == "" else f"{folder}/{filename}"
            filepath = quote(pathstr, safe="")
            return f"{self.base_url}/files/{filepath}?ref={ref}"

    def _decode_response(self, response, filename):
        json_response = json.loads(response.decode())
        content = base64.b64decode(json_response["content"].encode("ascii"))
        return RepositoryConnector._decode_response(self, content, filename)

    def search_files(self, folder="", suffix=None, recursive=False):
        page = 1
        results = []
        while True:
            loader = HttpRequest(
                self._build_url(folder, recursive=recursive, page=page),
                DECODERS[".json"],
            )
            results.extend(loader.data)
            if len(loader.data) < self._per_page:
                # no more pages
                break
            page += 1
        end = "" if suffix is None else suffix
        return [
            e["path"]
            for e in results
            if e["type"] == "blob" and e["name"].endswith(end)
        ]


class OwncloudConnector(RepositoryConnector):
    def __init__(self, server: str, share: int):
        RepositoryConnector.__init__(self, base_url=f"{server}/s/{share}")

    def search_files(self, folder="", suffix=None, recursive=False):
        raise NotImplementedError(
            f"File search in folders not implemented for {self.__class__.__name__}."
        )

    def _build_url(self, folder, filename):
        fpath = "" if folder == "" else f"path={quote(folder,safe='')}&"
        fpath += f"files={quote(filename)}"
        url = f"{self.base_url}/download?{fpath}"
        return url
