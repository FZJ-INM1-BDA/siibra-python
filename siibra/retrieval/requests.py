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
from .exceptions import EbrainsAuthenticationError
from ..commons import logger, HBP_AUTH_TOKEN, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET
from .. import __version__

import json
from zipfile import ZipFile
import requests
import os
from nibabel import Nifti1Image, GiftiImage, streamlines
from skimage import io
import gzip
from io import BytesIO
import urllib
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Callable, Any, TYPE_CHECKING
from enum import Enum
from functools import wraps
from time import sleep
import sys

if TYPE_CHECKING:
    from .repositories import GitlabConnector

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
    ".png": lambda b: io.imread(BytesIO(b)),
    ".npy": lambda b: np.load(BytesIO(b))
}


class SiibraHttpRequestError(Exception):

    def __init__(self, url: str, status_code: int, msg="Cannot execute http request."):
        self.url = url
        self.status_code = status_code
        self.msg = msg
        Exception.__init__(self)

    def __str__(self):
        return (
            f"{self.msg}\n\tStatus code:{self.status_code:68.68}\n\tUrl:{self.response.url:76.76}"
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

    def _retrieve(self, block_size=1024, min_bytesize_with_no_progress_info=2e8):
        # Loads the data from http if required.
        # If the data is already cached, None is returned,
        # otherwise data (as it is already in memory anyway).
        # The caller should load the cachefile only
        # if None is returned.

        if self.cached and not self.refresh:
            return
        else:
            # not yet in cache, perform http request.
            if self.msg_if_not_cached is not None:
                logger.debug(self.msg_if_not_cached)
            headers = self.kwargs.get('headers', {})
            other_kwargs = {key: self.kwargs[key] for key in self.kwargs if key != "headers"}
            if self.post:
                r = requests.post(self.url, headers={
                    **USER_AGENT_HEADER,
                    **headers,
                }, **other_kwargs, stream=True)
            else:
                r = requests.get(self.url, headers={
                    **USER_AGENT_HEADER,
                    **headers,
                }, **other_kwargs, stream=True)
            if r.ok:
                size_bytes = int(r.headers.get('content-length', 0))
                if size_bytes > min_bytesize_with_no_progress_info:
                    progress_bar = tqdm(
                        total=size_bytes, unit='iB', unit_scale=True,
                        position=0, leave=True,
                        desc=f"Downloading {os.path.split(self.url)[-1]} ({size_bytes / 1024**2:.1f} MiB)"
                    )
                temp_cachefile = self.cachefile + "_temp"
                with open(temp_cachefile, "wb") as f:
                    for data in r.iter_content(block_size):
                        if size_bytes > min_bytesize_with_no_progress_info:
                            progress_bar.update(len(data))
                        f.write(data)
                if size_bytes > min_bytesize_with_no_progress_info:
                    progress_bar.close()
                self.refresh = False
                os.rename(temp_cachefile, self.cachefile)
                with open(self.cachefile, 'rb') as f:
                    return f.read()
            else:
                raise SiibraHttpRequestError(status_code=r.status_code, url=self.url)

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
            except Exception:  # TODO: do not use bare except
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

    _KG_API_TOKEN: str = None
    _IAM_ENDPOINT: str = "https://iam.ebrains.eu/auth/realms/hbp"
    _IAM_DEVICE_ENDPOINT: str = None
    _IAM_DEVICE_MAXTRIES = 12
    _IAM_DEVICE_POLLING_INTERVAL_SEC = 5
    _IAM_DEVICE_FLOW_CLIENTID = "siibra"

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
    def init_oidc(cls):
        resp = requests.get(f"{cls._IAM_ENDPOINT}/.well-known/openid-configuration")
        json_resp = resp.json()
        if "token_endpoint" in json_resp:
            logger.debug(f"token_endpoint exists in .well-known/openid-configuration. Setting _IAM_TOKEN_ENDPOINT to {json_resp.get('token_endpoint')}")
            cls._IAM_TOKEN_ENDPOINT = json_resp.get("token_endpoint")
        else:
            logger.warn("expect token endpoint in .well-known/openid-configuration, but was not present")

        if "device_authorization_endpoint" in json_resp:
            logger.debug(f"device_authorization_endpoint exists in .well-known/openid-configuration. setting _IAM_DEVICE_ENDPOINT to {json_resp.get('device_authorization_endpoint')}")
            cls._IAM_DEVICE_ENDPOINT = json_resp.get("device_authorization_endpoint")
        else:
            logger.warn("expected device_authorization_endpoint in .well-known/openid-configuration, but was not present")

    @classmethod
    def fetch_token(cls):
        """Fetch an EBRAINS token using commandline-supplied username/password
        using the data proxy endpoint.
        :ref:`Details on how to access EBRAINS are here.<accessEBRAINS>`
        """
        cls.device_flow()
        
    @classmethod
    def device_flow(cls):
        if all([
            not sys.__stdout__.isatty(),  # if is tty, do not raise
            not any(k in ['JPY_INTERRUPT_EVENT', "JPY_PARENT_PID"] for k in os.environ),  # if is notebook environment, do not raise
            not os.getenv("SIIBRA_ENABLE_DEVICE_FLOW"),  # if explicitly enabled by env var, do not raise
        ]):
            raise EbrainsAuthenticationError(
                "sys.stdout is not tty, SIIBRA_ENABLE_DEVICE_FLOW is not set,"
                "and not running in a notebook. Are you running in batch mode?"
            )

        cls.init_oidc()
        resp = requests.post(
            url=cls._IAM_DEVICE_ENDPOINT,
            data={
                'client_id': cls._IAM_DEVICE_FLOW_CLIENTID
            }
        )
        resp.raise_for_status()
        resp_json = resp.json()
        logger.debug("device flow, request full json:", resp_json)

        assert "verification_uri_complete" in resp_json
        assert "device_code" in resp_json

        device_code = resp_json.get("device_code")

        print("***")
        print(f"To continue, please go to {resp_json.get('verification_uri_complete')}")
        print("***")

        attempt_number = 0
        sleep_timer = cls._IAM_DEVICE_POLLING_INTERVAL_SEC
        while True:
            # TODO the polling is a little busted at the moment.
            # need to speak to axel to shorten the polling duration
            sleep(sleep_timer)

            logger.debug("Calling endpoint")
            if attempt_number > cls._IAM_DEVICE_MAXTRIES:
                message = f"exceeded max attempts: {cls._IAM_DEVICE_MAXTRIES}, aborting..."
                logger.error(message)
                raise EbrainsAuthenticationError(message)
            attempt_number += 1
            resp = requests.post(
                url=cls._IAM_TOKEN_ENDPOINT,
                data={
                    'grant_type': "urn:ietf:params:oauth:grant-type:device_code",
                    'client_id': cls._IAM_DEVICE_FLOW_CLIENTID,
                    'device_code': device_code
                }
            )

            if resp.status_code == 200:
                json_resp = resp.json()
                logger.debug("Device flow sucessful:", json_resp)
                cls._KG_API_TOKEN = json_resp.get("access_token")
                print("ebrains token successfuly set.")
                break

            if resp.status_code == 400:
                json_resp = resp.json()
                error = json_resp.get("error")
                if error == "slow_down":
                    sleep_timer += 1
                logger.debug(f"400 error: {resp.content}")
                continue

            raise EbrainsAuthenticationError(resp.content)

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
        if HBP_AUTH_TOKEN:
            self.__class__._KG_API_TOKEN = HBP_AUTH_TOKEN
            return self.__class__._KG_API_TOKEN

        # try KEYCLOAK. Requires the following environment variables set:
        # KEYCLOAK_ENDPOINT, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET

        if KEYCLOAK_CLIENT_ID is not None and KEYCLOAK_CLIENT_SECRET is not None:
            logger.info("Getting an EBRAINS token via keycloak client configuration...")
            result = requests.post(
                self.__class__._IAM_TOKEN_ENDPOINT,
                data=(
                    f"grant_type=client_credentials&client_id={KEYCLOAK_CLIENT_ID}"
                    f"&client_secret={KEYCLOAK_CLIENT_SECRET}"
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
            if e.status_code in self.SC_MESSAGES:
                raise RuntimeError(self.SC_MESSAGES[e.status_code])
            else:
                raise RuntimeError(
                    f"Could not process HTTP request (status code: "
                    f"{e.status_code}). Message was: {e.msg}"
                    f"URL was: {e.url}"
                )
        return result


def try_all_connectors():
    def outer(fn):
        @wraps(fn)
        def inner(self: 'GitlabProxyEnum', *args, **kwargs):
            exceptions = []
            for connector in self.connectors:
                try:
                    return fn(self, *args, connector=connector, **kwargs)
                except Exception as e:
                    exceptions.append(e)
            else:
                for exc in exceptions:
                    logger.error(exc)
                raise Exception("try_all_connectors failed")
        return inner
    return outer


class GitlabProxyEnum(Enum):
    DATASET_V1 = "DATASET_V1"
    PARCELLATIONREGION_V1 = "PARCELLATIONREGION_V1"
    DATASET_V3 = "DATASET_V3"

    @property
    def connectors(self) -> List['GitlabConnector']:
        servers = [
            ("https://jugit.fz-juelich.de", 7846),
            ("https://gitlab.ebrains.eu", 421),
        ]
        from .repositories import GitlabConnector
        return [GitlabConnector(server[0], server[1], "master", archive_mode=True) for server in servers]

    @try_all_connectors()
    def search_files(self, folder: str, suffix=None, recursive=True, *, connector: 'GitlabConnector' = None) -> List[str]:
        assert connector
        return connector.search_files(folder, suffix=suffix, recursive=recursive)

    @try_all_connectors()
    def get(self, filename, decode_func=None, *, connector: 'GitlabConnector' = None):
        assert connector
        return connector.get(filename, "", decode_func)


class GitlabProxy(HttpRequest):

    folder_dict = {
        GitlabProxyEnum.DATASET_V1: "ebrainsquery/v1/datasets",
        GitlabProxyEnum.DATASET_V3: "ebrainsquery/v3/datasets",
        GitlabProxyEnum.PARCELLATIONREGION_V1: "ebrainsquery/v1/parcellationregions",
    }

    def __init__(
        self,
        flavour: GitlabProxyEnum,
        instance_id=None,
        postprocess: Callable[['GitlabProxy', Any], Any] = (
            lambda proxy, obj: obj
            if hasattr(proxy, "instance_id") and proxy.instance_id
            else {"results": obj}
        )
    ):
        if flavour not in GitlabProxyEnum:
            raise RuntimeError("Can only proxy enum members")

        self.flavour = flavour
        self.folder = self.folder_dict[flavour]
        self.postprocess = postprocess
        self.instance_id = instance_id
        self._cached_files = None

    def get(self):
        if self.instance_id:
            return self.postprocess(self, self.flavour.get(f"{self.folder}/{self.instance_id}.json"))
        return self.postprocess(self, self.flavour.get(f"{self.folder}/_all.json"))


class MultiSourceRequestException(Exception):
    pass


class MultiSourcedRequest:
    requests: List[HttpRequest] = []

    def __init__(self, requests: List[HttpRequest]) -> None:
        self.requests = requests

    def get(self):
        exceptions = []
        for req in self.requests:
            try:
                return req.get()
            except Exception as e:
                exceptions.append(e)
        else:
            raise MultiSourceRequestException("All requests failed:\n" + "\n".join(str(exc) for exc in exceptions))

    @property
    def data(self):
        return self.get()
