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
"""Connect to repositories to browse and pull files within."""

from .requests import (
    HttpRequest,
    EbrainsRequest,
    SiibraHttpRequestError,
    find_suitiable_decoder,
    DECODERS
)
from .cache import CACHE

from ..commons import logger, siibra_tqdm

from abc import ABC, abstractmethod
from urllib.parse import quote
import pathlib
import os
from zipfile import ZipFile
from typing import List


class RepositoryConnector(ABC):
    """
    Base class for repository connectors.
    """

    def __init__(self, base_url):
        self.base_url = base_url

    @abstractmethod
    def search_files(self, folder: str, suffix: str, recursive: bool = False) -> List[str]:
        """
        Get the files within the repository.

        Parameters
        ----------
        folder : str
            folder or folders in the form 'path/to/file'
        suffix : str
        recursive : bool, default: False
            If True, searches files in all subfolders

        Returns
        -------
        List[str]
            List of file names.
        """
        pass

    @abstractmethod
    def _build_url(self, folder: str, filename: str):
        pass

    def _decode_response(self, response, filename: str):
        decoder = find_suitiable_decoder(filename)
        return decoder(response) if decoder else response

    def get(self, filename, folder="", decode_func=None):
        """Get a file right away."""
        return self.get_loader(filename, folder, decode_func).data

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for executing the query
        only once loader.data is accessed."""
        url = self._build_url(folder, filename)
        if url is None:
            raise RuntimeError(f"Cannot build url for ({folder}, {filename})")
        if decode_func is None:
            return HttpRequest(url, lambda b: self._decode_response(b, filename))
        else:
            return HttpRequest(url, decode_func)

    def get_loaders(
        self, folder="", suffix=None, progress=None, recursive=False, decode_func=None
    ):
        """
        Returns an iterator with lazy loaders for the files in a given folder.
        In each iteration, a tuple (filename,file content) is returned.
        """
        fnames: List[str] = self.search_files(folder, suffix, recursive)
        result = [
            (fname, self.get_loader(fname, decode_func=decode_func)) for fname in fnames
        ]
        all_cached = all(_[1].cached for _ in result)
        if progress is None or all_cached:
            return result
        else:
            return list(siibra_tqdm(result, total=len(fnames), desc=progress))

    @classmethod
    def _from_url(cls, url: str):
        expurl = os.path.abspath(os.path.expanduser(url))
        if url.endswith(".zip"):
            return ZipfileConnector(url)
        elif os.path.isdir(expurl):
            return LocalFileRepository(expurl)
        else:
            raise TypeError(
                "Do not know how to create a repository "
                f"connector from url '{url}'."
            )

    def __eq__(self, other):
        return self.base_url == other.base_url


class LocalFileRepository(RepositoryConnector):

    def __init__(self, folder: str):
        self._folder = pathlib.Path(folder)
        assert pathlib.Path.is_dir(self._folder)

    def _build_url(self, folder: str, filename: str):
        return pathlib.Path.joinpath(self._folder, folder, filename)

    class FileLoader:
        """
        Just a loads a local file, but mimics the behaviour
        of cached http requests used in other connectors.
        """
        def __init__(self, file_url, decode_func):
            self.url = file_url
            self.func = decode_func
            self.cached = True

        @property
        def data(self):
            with open(self.url, 'rb') as f:
                return self.func(f.read())

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for loading data
        only once loader.data is accessed."""
        url = self._build_url(folder, filename)
        if url is None:
            raise RuntimeError(f"Cannot build url for ({folder}, {filename})")
        if decode_func is None:
            return self.FileLoader(url, lambda b: self._decode_response(b, filename))
        else:
            return self.FileLoader(url, decode_func)

    def search_files(self, folder="", suffix=None, recursive=False):
        results = []
        walk_pattern = f"{'**/' if recursive else ''}[!.~]*"
        for file in self._folder.joinpath(folder).glob(walk_pattern):
            if suffix is not None and not file.as_posix().endswith(suffix):
                continue
            results.append(file.relative_to(self._folder).as_posix())
        return results

    def __str__(self):
        return f"{self.__class__.__name__} at {self._folder}"

    def __eq__(self, other):
        return self._folder == other._folder


class GithubConnector(RepositoryConnector):

    def __init__(
        self,
        owner: str,
        repo: str,
        reftag: str,
        skip_branchtest=False,
        archive_mode=False
    ):
        """
        Connect to a GitHub repository with a specific ref (branch or tag).

        Parameters
        ----------
        owner : str
        repo : str
        reftag : str
            Tag or branch
        skip_branchtest : bool, default: False
            Whether to test if the reftag resides in the repository.
        archive_mode : bool, default: False
            Archive the repo (for reftag only) to siibra local cache.
        Raises
        ------
        RuntimeError
            If branch test could not find the reftag in the repo's list of tags
            and branches.
        """
        RepositoryConnector.__init__(
            self,
            base_url=f"https://api.github.com/repos/{owner}/{repo}"
        )
        assert reftag, "Please supply a branch name or tag for `reftag` to create a `GithubConnector`."
        if not skip_branchtest:
            try:
                tags = HttpRequest(f"{self.base_url}/tags", DECODERS[".json"], refresh=True).data
                branches = HttpRequest(f"{self.base_url}/branches", DECODERS[".json"], refresh=True).data
                matched_reftags = list(
                    filter(lambda b: b["name"] == reftag, tags + branches)
                )
                if len(matched_reftags) == 1:
                    self._want_commit_cached = matched_reftags[0]["commit"]
                else:
                    raise RuntimeError(f"Found {len(matched_reftags)} mathces to {reftag}")
                self._tag_checked = True
            except Exception:
                logger.warning("Could not connect to GitHub repository.", exc_info=1)
        self.reftag = reftag
        self._raw_baseurl = f"https://raw.githubusercontent.com/{owner}/{repo}/{self.reftag}"
        self.archive_mode = archive_mode
        self._archive_conn: LocalFileRepository = None
        self._recursed_tree = None

    def search_files(self, folder="", suffix="", recursive=False) -> List[str]:
        if self._recursed_tree is None:
            self._recursed_tree = HttpRequest(
                f"{self.base_url}/git/trees/{self.reftag}?recursive=1",
                DECODERS[".json"]
            ).data.get("tree", [])
        folder_depth = len(folder.split('/')) if folder else 0
        return [
            f["path"] for f in self._recursed_tree
            if f["type"] == "blob"
            and f["path"].startswith(folder)
            and f["path"].endswith(suffix)
            and (recursive or len(f["path"].split('/')) == folder_depth + 1)
        ]

    def _build_url(self, folder: str, filename: str):
        pathstr = pathlib.Path(folder, filename or "").as_posix()
        return f'{self._raw_baseurl}/{quote(pathstr, safe="")}'

    def get_loader(self, filename, folder="", decode_func=None):
        if self.archive_mode:
            self._archive()
            return self._archive_conn.get_loader(filename, folder, decode_func)
        else:
            return super().get_loader(filename, folder, decode_func)

    def _archive(self):
        assert self.archive_mode, "To archive the repo, `archive_mode` must be True."
        archive_directory = CACHE.build_filename(self.base_url + self.reftag)
        if not os.path.isdir(archive_directory):
            import tarfile

            tarball_url = f"{self.base_url}/tarball/{self.reftag}"
            req = HttpRequest(tarball_url, func=lambda b: b)
            req.get()
            with tarfile.open(name=req.cachefile, mode="r:gz") as tar:
                tar.extractall(CACHE.folder)
                foldername = tar.getnames()[0]
            os.rename(os.path.join(CACHE.folder, foldername), archive_directory)
        if self._archive_conn is None:
            # create LocalFileRepository as an interface to the local files
            self._archive_conn = LocalFileRepository(archive_directory)


class GitlabConnector(RepositoryConnector):

    def __init__(self, server: str, project: int, reftag: str, skip_branchtest=False, *, archive_mode=False):
        """
        archive_mode: in archive mode, the entire repository is downloaded as an archive. This is necessary/could be useful for repositories with numerous files.
            n.b. only archive_mode should only be set for trusted domains. Extraction of archive can result in files created outside the path
            see https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        """
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
        self.archive_mode = archive_mode
        self._archive_conn: LocalFileRepository = None

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
                logger.warning("Could not connect to gitlab server!")
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
            return f"{self.base_url}/files/{filepath}/raw?ref={ref}"

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

    def get_loader(self, filename, folder="", decode_func=None):
        if self.archive_mode:
            self._archive()
            return self._archive_conn.get_loader(filename, folder, decode_func)
        else:
            return super().get_loader(filename, folder, decode_func)

    def _archive(self):
        assert self.archive_mode, "To archive the repo, `archive_mode` must be True."
        ref = self.reftag if self.want_commit is None else self.want_commit["short_id"]
        archive_directory = CACHE.build_filename(self.base_url + ref)
        if not os.path.isdir(archive_directory):
            import tarfile

            tarball_url = self.base_url + f"/archive.tar.gz?sha={ref}"
            req = HttpRequest(tarball_url, func=lambda b: b)
            req.get()
            with tarfile.open(name=req.cachefile, mode="r:gz") as tar:
                tar.extractall(CACHE.folder)
                foldername = tar.getnames()[0]
            os.rename(os.path.join(CACHE.folder, foldername), archive_directory)
        if self._archive_conn is None:
            # create LocalFileRepository as an interface to the local files
            self._archive_conn = LocalFileRepository(archive_directory)

    def __eq__(self, other):
        return all([
            self.base_url == other.base_url,
            self.reftag == other.reftag
        ])


class ZipfileConnector(RepositoryConnector):

    def __init__(self, url: str):
        RepositoryConnector.__init__(self, base_url="")
        self.url = url
        self._zipfile_cached = None

    @property
    def zipfile(self):
        if self._zipfile_cached is None:
            if os.path.isfile(os.path.abspath(os.path.expanduser(self.url))):
                self._zipfile_cached = os.path.abspath(os.path.expanduser(self.url))
            else:
                # assume the url is web URL to download the zip!
                req = HttpRequest(self.url)
                req._retrieve()
                self._zipfile_cached = req.cachefile
        return self._zipfile_cached

    def _build_url(self, folder="", filename=None):
        return os.path.join(folder, filename)

    def search_files(self, folder="", suffix="", recursive=False):
        container = ZipFile(self.zipfile)
        result = []
        if folder and not folder.endswith(os.path.sep):
            folder += os.path.sep
        for fname in container.namelist():
            if os.path.dirname(fname.replace(folder, "")) and not recursive:
                continue
            if not os.path.basename(fname):
                continue
            if fname.startswith(folder) and fname.endswith(suffix):
                result.append(fname)
        return result

    def __eq__(self, other):
        return self.url == other.url

    def clear_cache(self):
        os.remove(self.zipfile)
        self._zipfile_cached = None

    class ZipFileLoader:
        """
        Loads a file from the zip archive, but mimics the behaviour
        of cached http requests used in other connectors.
        """
        def __init__(self, zipfile, filename, decode_func):
            self.zipfile = zipfile
            self.filename = filename
            self.func = decode_func
            self.cachefile = CACHE.build_filename(zipfile + filename)

        @property
        def cached(self):
            return os.path.isfile(self.cachefile)

        @property
        def data(self):
            container = ZipFile(self.zipfile)
            return self.func(container.open(self.filename).read())

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for loading data
        only once loader.data is accessed."""
        if decode_func is None:
            return self.ZipFileLoader(self.zipfile, filename, lambda b: self._decode_response(b, filename))
        else:
            return self.ZipFileLoader(self.zipfile, filename, decode_func)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.zipfile}"


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


class EbrainsHdgConnector(RepositoryConnector):
    """Download sensitive files from EBRAINS using
    the Human Data Gateway (HDG) via the data proxy API.
    Service documentation can be found here https://data-proxy.ebrains.eu/api/docs
    """

    """
    Version of the data-proxy API that should be used for this request.
    Currently v1 is the only supported version."""
    api_version = "v1"

    """
    Base URL for the Dataset Endpoint of the Data-Proxy API
    https://data-proxy.ebrains.eu/api/docs#/datasets

    Supported functions by the endpoint:
    ------------------------------------
    - POST: Request access to the dataset.
        This is required for the other functions.
    - GET: Return list of all available objects in the dataset
    """
    base_url = f"https://data-proxy.ebrains.eu/api/{api_version}/datasets"

    """
    Limit of returned objects
    Default value on API side is 50 objects
    """
    maxentries = 1000

    def __init__(self, dataset_id):
        """Construct a dataset query for the Human Data Gateway.

        Parameters
        ----------
        dataset_id : str
            EBRAINS dataset id for a dataset that is exposed
            via the human data gateway.
        """

        self._files = []
        self.dataset_id = dataset_id

        marker = None
        while True:

            # The endpoint implements basic pagination, using the filenames as markers.

            if marker is None:
                url = f"{self.base_url}/{dataset_id}?limit={self.maxentries}"
            else:
                url = f"{self.base_url}/{dataset_id}?limit={self.maxentries}&marker={marker}"

            try:
                result = EbrainsRequest(url, DECODERS[".json"]).get()
            except SiibraHttpRequestError as e:
                if e.status_code in [401, 422]:
                    # Request access to the dataset (401: expired, 422: not yet requested)
                    EbrainsRequest(f"{self.base_url}/{dataset_id}", post=True).get()
                    input(
                        "You should have received an email with a confirmation link - "
                        "please find that email and click on the link, then press enter "
                        "to continue"
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"Could not request private file links for dataset {dataset_id}. "
                        f"Status code was: {e.response.status_code}. "
                        f"Message was: {e.response.text}. "
                    )

            newfiles = result["objects"]
            self._files.extend(newfiles)
            logger.debug(f"{len(newfiles)} of {self.maxentries} objects returned.")

            if len(newfiles) == self.maxentries:
                # there might be more files
                marker = newfiles[-1]["name"]
            else:
                logger.info(
                    f"{len(self._files)} objects found for dataset {dataset_id} returned."
                )
                self.container = result["container"]
                self.prefix = result["prefix"]
                break

    def search_files(self, folder="", suffix=None, recursive=False):
        result = []
        for f in self._files:
            if f["name"].startswith(folder):
                if suffix is None:
                    result.append(f["name"])
                else:
                    if f["name"].endswith(suffix):
                        result.append(f["name"])
        return result

    def _build_url(self, folder, filename):
        if len(folder) > 0:
            fpath = quote(f"{folder}/{filename}")
        else:
            fpath = quote(f"{filename}")
        url = f"{self.base_url}/{self.dataset_id}/{fpath}?redirect=true"
        return url

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for executing the query
        only once loader.data is accessed."""
        return EbrainsRequest(self._build_url(folder, filename), decode_func)


class EbrainsPublicDatasetConnector(RepositoryConnector):
    """Access files from public EBRAINS datasets via the Knowledge Graph v3 API."""

    QUERY_ID = "bebbe365-a0d6-41ea-9ff8-2554c15f70b7"
    base_url = "https://core.kg.ebrains.eu/v3-beta/queries/"
    maxentries = 1000

    def __init__(self, dataset_id: str = None, version_id: str = None, title: str = None, in_progress=False):
        """Construct a dataset query with the dataset id.

        Parameters
        ----------
        dataset_id : str
            EBRAINS dataset id of a public dataset in KG v3.
        version_id : str
            Version id to pick from the dataset (optional)
        title: str
            Part of dataset title as an alternative dataset specification (will ignore dataset_id then)
        in_progress: bool (default:False)
            If true, will request datasets that are still under curation.
            Will only work when autenticated with an appropriately privileged
            user account.
        """
        self.dataset_id = dataset_id
        self.versions = {}
        self._description = ""
        self._name = ""
        self.use_version = None

        stage = "IN_PROGRESS" if in_progress else "RELEASED"
        if title is None:
            assert dataset_id is not None
            self.dataset_id = dataset_id
            url = f"{self.base_url}/{self.QUERY_ID}/instances?stage={stage}&dataset_id={dataset_id}"
        else:
            assert dataset_id is None
            logger.info(f"Using title '{title}' for EBRAINS dataset search, ignoring id '{dataset_id}'")
            url = f"{self.base_url}/{self.QUERY_ID}/instances?stage={stage}&title={title}"

        response = EbrainsRequest(url, DECODERS[".json"]).get()
        results = response.get('data', [])
        if len(results) != 1:
            if dataset_id is None:
                for r in results:
                    print(r['name'])
                    raise RuntimeError(f"Search for '{title}' yielded {len(results)} datasets. Please refine your specification.")
            else:
                raise RuntimeError(f"Dataset id {dataset_id} did not yield a unique match, please fix the dataset specification.")

        data = results[0]
        self.id = data['id']
        if title is not None:
            self.dataset_id = data['id']
        self._description += data.get("description", "")
        self._name += data.get("name", "")
        self.versions = {v["versionIdentifier"]: v for v in data["versions"]}
        if version_id is None:
            self.use_version = sorted(list(self.versions.keys()))[-1]
            if len(self.versions) > 1:
                logger.info(
                    f"Found {len(self.versions)} versions for dataset '{data['name']}' "
                    f"({', '.join(self.versions.keys())}). "
                    f"Will use {self.use_version} per default."
                )
        else:
            assert version_id in self.versions
            self.use_version = version_id

    @property
    def name(self):
        if self.use_version in self.versions:
            if "name" in self.versions[self.use_version]:
                if len(self.versions[self.use_version]["name"]) > 0:
                    return self.versions[self.use_version]["name"]
        return self._name

    @property
    def description(self):
        result = self._description
        if self.use_version in self.versions:
            result += "\n" + self.versions[self.use_version].get("description", "")
        return result

    @property
    def authors(self):
        result = []
        if self.use_version in self.versions:
            for author_info in self.versions[self.use_version]["authors"]:
                result.append(f"{author_info['familyName']}, {author_info['givenName']}")
        return result

    @property
    def citation(self):
        if self.use_version in self.versions:
            return self.versions[self.use_version].get("cite", "")
        else:
            return None

    @property
    def _files(self):
        if self.use_version in self.versions:
            return {
                f["name"]: f["url"] for f in self.versions[self.use_version]["files"]
            }
        else:
            return {}

    def search_files(self, folder="", suffix=None, recursive=False):
        result = []
        for fname in self._files:
            if fname.startswith(folder):
                if suffix is None:
                    result.append(fname)
                else:
                    if fname.endswith(suffix):
                        result.append(fname)
        return result

    def _build_url(self, folder, filename):
        fpath = f"{folder}/{filename}" if len(folder) > 0 else f"{filename}"
        if fpath not in self._files:
            raise RuntimeError(
                f"The file {fpath} requested from EBRAINS dataset {self.dataset_id} is not available in this repository."
            )
        return self._files[fpath]

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for executing the query
        only once loader.data is accessed."""
        return HttpRequest(self._build_url(folder, filename), decode_func)


class EbrainsPublicDatasetConnectorMinds(RepositoryConnector):
    """Access files from public EBRAINS datasets via the Knowledge Graph v3 API."""

    QUERY_ID = "siibra-minds-dataset-v1"
    base_url = "https://kg.humanbrainproject.eu/query/minds/core/dataset/v1.0.0"
    maxentries = 1000

    def __init__(self, dataset_id=None, title=None, in_progress=False):
        """Construct a dataset query with the dataset id.

        Parameters
        ----------
        dataset_id : str
            EBRAINS dataset id of a public dataset in KG v3.
        title: str
            Part of dataset title as an alternative dataset specification (will ignore dataset_id then)
        in_progress: bool, default: False
            If true, will request datasets that are still under curation.
            Will only work when authenticated with an appropriately privileged
            user account.
        """
        stage = "IN_PROGRESS" if in_progress else "RELEASED"
        if title is None:
            assert dataset_id is not None
            self.dataset_id = dataset_id
            url = f"{self.base_url}/{self.QUERY_ID}/instances?databaseScope={stage}&dataset_id={dataset_id}"
        else:
            assert dataset_id is None
            logger.info(f"Using title '{title}' for EBRAINS dataset search, ignoring id '{dataset_id}'")
            url = f"{self.base_url}/{self.QUERY_ID}/instances?databaseScope={stage}&title={title}"
        req = EbrainsRequest(url, DECODERS[".json"])
        response = req.get()
        self._files = {}
        results = response.get('results', [])
        if dataset_id is not None:
            assert len(results) < 2
        elif len(results) > 1:
            for r in results:
                print(r.keys())
                print(r['name'])
            raise RuntimeError(f"Search for '{title}' yielded {len(results)} datasets, see above. Please refine your specification.")
        for res in results:
            if title is not None:
                self.dataset_id = res['id']
            self.id = res['id']
            for fileinfo in res['https://schema.hbp.eu/myQuery/v1.0.0']:
                self._files[fileinfo['relative_path']] = fileinfo['path']

    def search_files(self, folder="", suffix=None, recursive=False):
        result = []
        for fname in self._files:
            if fname.startswith(folder):
                if suffix is None:
                    result.append(fname)
                else:
                    if fname.endswith(suffix):
                        result.append(fname)
        return result

    def _build_url(self, folder, filename):
        fpath = f"{folder}/{filename}" if len(folder) > 0 else f"{filename}"
        if fpath not in self._files:
            raise RuntimeError(
                f"The file {fpath} requested from EBRAINS dataset {self.dataset_id} is not available in this repository."
            )
        return self._files[fpath]

    def get_loader(self, filename, folder="", decode_func=None):
        """Get a lazy loader for a file, for executing the query
        only once loader.data is accessed."""
        return HttpRequest(self._build_url(folder, filename), decode_func)
