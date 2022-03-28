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

from .requests import DECODERS, HttpRequest, EbrainsRequest, SiibraHttpRequestError
from .. import logger
from abc import ABC, abstractmethod
from urllib.parse import quote
from tqdm import tqdm
import getpass
import ebrains_drive
import json
import os
from io import StringIO
from uuid import UUID

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
        fnames = self.search_files(folder, suffix, recursive)
        result = [
            (fname, self.get_loader(fname, decode_func=decode_func)) for fname in fnames
        ]
        all_cached = all(_[1].cached for _ in result)
        if progress is None or all_cached:
            return result
        else:
            return tqdm(
                result, total=len(fnames), desc=progress, disable=logger.level > 20
            )


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
            pathstr = "" if len(folder) == 0 else f"&path={quote(folder, safe='')}"
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


class OwncloudConnector(RepositoryConnector):
    def __init__(self, server: str, share: int):
        RepositoryConnector.__init__(self, base_url=f"{server}/s/{share}")

    def search_files(self, folder="", suffix=None, recursive=False):
        raise NotImplementedError(
            f"File search in folders not implemented for {self.__class__.__name__}."
        )

    def _build_url(self, folder, filename):
        fpath = "" if folder == "" else f"path={quote(folder, safe='')}&"
        fpath += f"files={quote(filename)}"
        url = f"{self.base_url}/download?{fpath}"
        return url


class EbrainsDriveConnector(RepositoryConnector):

    connected = False
    default_repo = None
    drive_dir = None

    def __init__(self, folder='Siibra annotations', auth_token=None, username=None, password=None):
        """
        :param folder: str (default: 'Siibra annotations')
            Ebrains drive folder name
        :param auth_token: str (optional)
            Auth token should contain Ebrains drive scope
        :param username: str (optional)
            Ebrains username
        :param password: str (optional)
            Ebrains password
        """
        if not auth_token or not (username and password):
            user_action = input('Kg token or username/password is not defined. \n Press:'
                                ' \n 1. To input access token \n 2. To input username and password')

            if user_action == '1':
                auth_token = input('Input access token: \n')
            elif user_action == '2':
                username = input('Username: \n')
                password = getpass.getpass('Password: \n')
            else:
                raise Exception('Authentication credentials are mandatory to connect ebrains drive')

        self.folder = folder

        self.connect(auth_token, username, password)

    def connect(self, auth_token=None, username=None, password=None):
        """
        :param auth_token: str (optional)
            Auth token should contain Ebrains drive scope
        :param username: str (optional)
            Ebrains username
        :param password: str (optional)
            Ebrains password
        """
        assert auth_token or (username and password)

        if auth_token:
            client = ebrains_drive.connect(token=auth_token)
        if username and password:
            client = ebrains_drive.connect(username=username, password=password)

        if not client:
            print('Credentials invalid')

        list_repos = client.repos.list_repos()
        self.default_repo = client.repos.get_repo(list_repos[0].id)

        root_dir = self.default_repo.get_dir('/')
        if not root_dir.check_exists(self.folder):
            root_dir.mkdir(self.folder)

        self.drive_dir = self.default_repo.get_dir(f'/{self.folder}')
        self.connected = True

    # ToDo Question `@id` and `@type` filters are too specific??
    def search_files(self, id_f=None, type_f=None):
        """
        :param id_f: str (optional)
            Returns specific file content by '@id'
        :param type_f:  str (optional)
            Filters files by '@type' parameter
        :return: list
            Returns list of the file contents
        """
        if not self.connected:
            self.connect()

        stored_files = [f for f in self.drive_dir.ls()
                        if isinstance(f, ebrains_drive.files.SeafFile)]

        if not stored_files:
            return

        files = []
        for file in stored_files:
            file = self.default_repo.get_file(file.path)
            obj = json.loads(file.get_content())
            obj['name'] = os.path.splitext(file.name)[0]
            if id_f:
                if obj['@id'] == id_f:
                    return obj
            else:
                files.append(obj)

        if type_f:
            files = list(filter(lambda c: c['@type'] == type_f, files))

        return files

    def _build_url(self, folder: str, filename: str):
        pass

    def save(self, obj, name='Unnamed'):
        """
        :param obj: JSON object
            file content
        :param name: (default: unnamed)
        """
        if not self.connected:
            self.connect()

        ##### ToDo check if file exists with same id and remove?
        #      Or check if file exists with same id and name and then remove?
        #      Or do not remove at all?
        # if self.drive_id and self.file_name and (not name or self.file_name == name):
        #     self.remove_from_drive()

        if not name:
            name = 'Unnamed'

        file = StringIO(json.dumps(json.loads(obj), indent=4, sort_keys=True, cls=self.UUIDEncoder))
        self.drive_dir.upload(file, f'{name}.json')

    def remove(self, id_f):
        """
        :param id_f: str
             Removes file by '@id'
        """
        assert id_f
        if self.connected is False:
            self.connect()

        files = [file for file in self.drive_dir.ls()
                 if isinstance(file, ebrains_drive.files.SeafFile)]

        for file in files:
            stored_file = self.default_repo.get_file(file.path)
            file_id = json.loads(stored_file.get_content())['@id']
            if file_id == id_f:
                stored_file.delete()
                return

        print('File not found')

    class UUIDEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, UUID):
                return obj.hex
            return json.JSONEncoder.default(self, obj)

class EbrainsHdgConnector(RepositoryConnector):
    """Download sensitive files from EBRAINS using
    the Human Data Gateway (HDG) via the data proxy API.
    """

    base_url = "https://data-proxy.ebrains.eu/api/datasets"
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
                if e.response.status_code in [401, 422]:
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
            fpath = quote(f"{folder}/{filename}", safe="")
        else:
            fpath = quote(f"{filename}", safe="")
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

    def __init__(self, dataset_id, in_progress=False):
        """Construct a dataset query with the dataset id.

        Parameters
        ----------
        dataset_id : str
            EBRAINS dataset id of a public dataset in KG v3.
        in_progress: bool (default:False)
            If true, will request datasets that are still under curation.
            Will only work when autenticated with an appropriately privileged
            user account.
        """
        self.dataset_id = dataset_id
        stage = "IN_PROGRESS" if in_progress else "RELEASED"
        url = f"{self.base_url}/{self.QUERY_ID}/instances?stage={stage}&dataset_id={dataset_id}"
        result = EbrainsRequest(url, DECODERS[".json"]).get()
        self.versions = {}
        self._description = ""
        self._name = ""
        self.use_version = None
        assert len(result["data"]) < 2
        if len(result["data"]) == 1:
            data = result["data"][0]
            self._description += data.get("description", "")
            self._name += data.get("name", "")
            self.versions = {v["versionIdentifier"]: v for v in data["versions"]}
            self.use_version = sorted(list(self.versions.keys()))[-1]
            if len(self.versions) > 1:
                logger.info(
                    f"Found {len(self.versions)} versions for dataset '{data['name']}' "
                    f"({', '.join(self.versions.keys())}). "
                    f"Will use {self.use_version} per default."
                )

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
