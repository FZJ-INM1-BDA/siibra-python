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

from .cache import CACHE
from ..commons import logger
from .. import __version__

import json
from zipfile import ZipFile
import requests
import os
from nibabel import Nifti1Image, GiftiImage, streamlines
from skimage import io
import gzip
from getpass import getpass
from io import BytesIO
import urllib
import pandas as pd

USER_AGENT_HEADER = {"User-Agent": f"siibra-python/{__version__}"}

DECODERS = {
    ".nii.gz": lambda b: Nifti1Image.from_bytes(gzip.decompress(b)),
    ".nii": lambda b: Nifti1Image.from_bytes(b),
    ".gii": lambda b: GiftiImage.from_bytes(b),
    ".json": lambda b: json.loads(b.decode()),
    ".tck": lambda b: streamlines.load(BytesIO(b)),
    ".csv": lambda b: pd.read_csv(BytesIO(b), delimiter=";"),
    ".tsv": lambda b: pd.read_csv(BytesIO(b), delimiter="\t"),
    ".txt": lambda b: pd.read_csv(BytesIO(b), delimiter=" ", header=None),
    ".zip": lambda b: ZipFile(BytesIO(b)),
    ".png": lambda b: io.imread(BytesIO(b))
}


class SiibraHttpRequestError(Exception):
    def __init__(self, response, msg="Cannot execute http request."):
        self.response = response
        self.msg = msg
        Exception.__init__(self)

    def __str__(self):
        return (
            f"{self.msg}\n"
            f"    Status code: {self.response.status_code}\n"
            f"    Url:         {self.response.url}\n"
        )


class HttpRequest:
    def __init__(
        self,
        url,
        func=None,
        msg_if_not_cached=None,
        refresh=False,
        post=False,
        **kwargs,
    ):
        """
        Initialize a cached http data loader.
        It takes a URL and optional data conversion function.
        For loading, the http request is only performed if the
        result is not yet available in the disk cache.
        Leaves the interpretation of the returned content to the caller.

        Parameters
        ----------
        url : string, or None
            URL for loading raw data, which is then fed into `func`
            for creating the output.
            If None, `func` will be called without arguments.
        func : function pointer
            Function for constructing the output data
            (called on the data retrieved from `url`, if supplied)
        refresh : bool, default: False
            If True, a possibly cached content will be ignored and refreshed
        post: bool, default: False
            perform a post instead of get
        """
        assert url is not None
        self.url = url
        suitable_decoders = [dec for sfx, dec in DECODERS.items() if url.endswith(sfx)]
        if (func is None) and (len(suitable_decoders) > 0):
            assert len(suitable_decoders) == 1
            self.func = suitable_decoders[0]
        else:
            self.func = func
        self.kwargs = kwargs
        self.cachefile = CACHE.build_filename(self.url + json.dumps(kwargs))
        self.msg_if_not_cached = msg_if_not_cached
        self.refresh = refresh
        self.post = post
        self._set_decoder_func(func, url)

    def _set_decoder_func(self, func, fileurl: str):
        urlpath = urllib.parse.urlsplit(fileurl).path
        if func is None:
            suitable_decoders = [
                dec for sfx, dec in DECODERS.items() if urlpath.endswith(sfx)
            ]
            if len(suitable_decoders) > 0:
                assert len(suitable_decoders) == 1
                self.func = suitable_decoders[0]
                return
        self.func = func

    @property
    def cached(self):
        return os.path.isfile(self.cachefile)

    def _retrieve(self):
        # Loads the data from http if required.
        # If the data is already cached, None is returned,
        # otherwise data (as it is already in memory anyway).
        # The caller should load the cachefile only
        # if None is returned.

        if self.cached and not self.refresh:
            # in cache. Just load the file
            logger.debug(
                f"Already in cache at {os.path.basename(self.cachefile)}: {self.url}"
            )
            return
        else:
            # not yet in cache, perform http request.
            logger.debug(f"Loading {self.url} to {os.path.basename(self.cachefile)}")
            if self.msg_if_not_cached is not None:
                logger.info(self.msg_if_not_cached)
            headers = self.kwargs.get('headers', {})
            other_kwargs = {key: self.kwargs[key] for key in self.kwargs if key != "headers"}
            if self.post:
                r = requests.post(self.url, headers={
                    **USER_AGENT_HEADER,
                    **headers,
                }, **other_kwargs)
            else:
                r = requests.get(self.url, headers={
                    **USER_AGENT_HEADER,
                    **headers,
                }, **other_kwargs)
            if r.ok:
                with open(self.cachefile, "wb") as f:
                    f.write(r.content)
                self.refresh = False
                return r.content
            else:
                raise SiibraHttpRequestError(r)

    def get(self):
        data = self._retrieve()
        if data is None:
            with open(self.cachefile, "rb") as f:
                data = f.read()
        try:
            return data if self.func is None else self.func(data)
        except Exception as e:
            # if network error results in bad cache, it may get raised here
            # e.g. BadZipFile("File is not a zip file")
            # if that happens, remove cachefile and
            try:
                os.unlink(self.cachefile)
            except:
                pass
            raise e

    @property
    def data(self):
        # for backward compatibility with old LazyHttpRequest class
        return self.get()


class ZipfileRequest(HttpRequest):
    def __init__(self, url, filename, func=None):
        HttpRequest.__init__(self, url, func=func)
        self.filename = filename
        self._set_decoder_func(func, filename)

    def get(self):
        self._retrieve()
        zipfile = ZipFile(self.cachefile)
        filenames = zipfile.namelist()
        matches = [fn for fn in filenames if fn.endswith(self.filename)]
        if len(matches) == 0:
            raise RuntimeError(
                f"Requested filename {self.filename} not found in archive at {self.url}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f'Requested filename {self.filename} was not unique in archive at {self.url}. Candidates were: {", ".join(matches)}'
            )
        with zipfile.open(matches[0]) as f:
            data = f.read()
        return data if self.func is None else self.func(data)


class EbrainsRequest(HttpRequest):
    """
    Implements lazy loading of HTTP Knowledge graph queries.
    """

    _KG_API_TOKEN = None
    keycloak_endpoint = (
        "https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect/token"
    )

    def __init__(
        self, url, decoder=None, params={}, msg_if_not_cached=None, post=False
    ):
        """Construct an EBRAINS request."""
        # NOTE: we do not pass params and header here,
        # since we want to evaluate them late in the get() method.
        # This is nice because it allows to set env. variable KG_TOKEN only when
        # really needed, and not necessarily on package initialization.
        self.params = params
        HttpRequest.__init__(self, url, decoder, msg_if_not_cached, post=post)

    @classmethod
    def fetch_token(cls):
        """Fetch an EBRAINS token using commandline-supplied username/password
        using the data proxy endpoint.
        """
        username = input("Your EBRAINS username: ")
        password = getpass("Your EBRAINS password: ")
        response = requests.post(
            "https://data-proxy.ebrains.eu/api/auth/token",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                **USER_AGENT_HEADER,
            },
            data=f'{{"username": "{username}", "password": "{password}"}}',
        )
        if response.status_code == 200:
            cls._KG_API_TOKEN = response.json()
        else:
            if response.status_code == 500:
                logger.error(
                    "Invalid EBRAINS username/password provided for fetching token."
                )
            raise SiibraHttpRequestError(response)

    @classmethod
    def set_token(cls, token):
        logger.info(f"Setting EBRAINS Knowledge Graph authentication token: {token}")
        cls._KG_API_TOKEN = token

    @property
    def kg_token(self):

        # token is available, return it
        if self.__class__._KG_API_TOKEN is not None:
            return self.__class__._KG_API_TOKEN

        # See if a token is directly provided in  $HBP_AUTH_TOKEN
        if "HBP_AUTH_TOKEN" in os.environ:
            self.__class__._KG_API_TOKEN = os.environ["HBP_AUTH_TOKEN"]
            return self.__class__._KG_API_TOKEN

        # try KEYCLOAK. Requires the following environment variables set:
        # KEYCLOAK_ENDPOINT, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET
        keycloak = {
            v: os.environ.get(f"KEYCLOAK_{v.upper()}")
            for v in ["client_id", "client_secret"]
        }
        if None not in keycloak.values():
            logger.info("Getting an EBRAINS token via keycloak client configuration...")
            result = requests.post(
                self.__class__.keycloak_endpoint,
                data=(
                    f"grant_type=client_credentials&client_id={keycloak['client_id']}"
                    f"&client_secret={keycloak['client_secret']}"
                    "&scope=kg-nexus-role-mapping%20kg-nexus-service-account-mock"
                ),
                headers={
                    "content-type": "application/x-www-form-urlencoded",
                    **USER_AGENT_HEADER,
                },
            )
            try:
                content = json.loads(result.content.decode("utf-8"))
            except json.JSONDecodeError as error:
                logger.error(f"Invalid json from keycloak:{error}")
                self.__class__._KG_API_TOKEN = None
            if "error" in content:
                logger.error(content["error_description"])
                self.__class__._KG_API_TOKEN = None
            self.__class__._KG_API_TOKEN = content["access_token"]

        if self.__class__._KG_API_TOKEN is None:
            # No success getting the token
            raise RuntimeError(
                "No access token for EBRAINS Knowledge Graph found. "
                "If you do not have an EBRAINS account, please first register at "
                "https://ebrains.eu/register. Then, use one of the following option: "
                "\n 1. Let siibra get you a token by passing your username and password, using siibra.fetch_ebrains_token()"
                "\n 2. If you know how to get a token yourself, set it as $HBP_AUTH_TOKEN or siibra.set_ebrains_token()"
                "\n 3. If you are an application developer, you might configure keycloak access by setting $KEYCLOAK_ENDPOINT, "
                "$KEYCLOAK_CLIENT_ID and $KEYCLOAK_CLIENT_SECRET."
            )

        return self.__class__._KG_API_TOKEN

    @property
    def auth_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.kg_token}",
        }

    def get(self):
        """Evaluate KG Token is evaluated only on execution of the request."""
        self.kwargs = {"headers": self.auth_headers, "params": self.params}
        return super().get()


class EbrainsKgQuery(EbrainsRequest):
    """Request outputs from a knowledge graph query."""

    server = "https://kg.humanbrainproject.eu"
    org = "minds"
    domain = "core"
    version = "v1.0.0"

    SC_MESSAGES = {
        401: "The provided EBRAINS authentication token is not valid",
        403: "No permission to access the given query",
        404: "Query with this id not found",
    }

    def __init__(self, query_id, instance_id=None, schema="dataset", params={}):
        inst_tail = "/" + instance_id if instance_id is not None else ""
        self.schema = schema
        url = "{}/query/{}/{}/{}/{}/{}/instances{}?databaseScope=RELEASED".format(
            self.server,
            self.org,
            self.domain,
            self.schema,
            self.version,
            query_id,
            inst_tail,
        )
        EbrainsRequest.__init__(
            self,
            url,
            decoder=DECODERS[".json"],
            params=params,
            msg_if_not_cached=f"Executing EBRAINS KG query {query_id}{inst_tail}",
        )

    def get(self):
        try:
            result = EbrainsRequest.get(self)
        except SiibraHttpRequestError as e:
            if e.response.status_code in self.SC_MESSAGES:
                raise RuntimeError(self.SC_MESSAGES[e.response.status_code])
            else:
                raise RuntimeError(
                    f"Could not process HTTP request (status code: "
                    f"{e.response.status_code}). Message was: {e.msg}"
                    f"URL was: {e.response.url}"
                )
        return result
