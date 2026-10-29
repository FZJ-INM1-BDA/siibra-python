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
"""Request files with decoders, lazy loading, and caching."""

import json
from zipfile import ZipFile
import requests
import os
import shutil
from pathlib import Path
import urllib.parse
from typing import List, Callable, TYPE_CHECKING, Literal, Optional, Union
from enum import Enum
from functools import wraps
from time import sleep
from dataclasses import dataclass, field, replace as replace_dataclass_fields

import numpy as np
import pandas as pd
from skimage import io as skimage_io
from nibabel import load as load_nibabel, streamlines, freesurfer
import h5py

from . import exceptions as _exceptions
from .cache import CACHE, cache_user_fn
from .. import __version__
from ..commons import (
    logger,
    HBP_AUTH_TOKEN,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET,
    siibra_tqdm,
    SIIBRA_USE_LOCAL_SNAPSPOT,
)

try:
    from trx.trx_file_memmap import load as _load_trx
    _HAS_TRX = True
except ImportError:
    _HAS_TRX = False
    logger.info("trx-python is not available. Requires python>=3.11.")

if TYPE_CHECKING:
    from .repositories import GitlabConnector

USER_AGENT_HEADER = {"User-Agent": f"siibra-python/{__version__}"}
_SESSION = requests.Session()


def _get_suffix(filename: str) -> Optional[str]:
    """
    Return the meaningful file suffix to preserve in the cache.

    Compound gzip suffixes are preserved, e.g. ``.nii.gz`` or ``.csv.gz``.
    For non-gzipped files, only the final suffix is returned.

    When the URL path itself has no suffix (e.g. the filename is embedded in
    a query parameter such as ``?prefix=path/to/file.trx``), query parameter
    values are scanned recursively. This handles nested URLs where the actual
    filename appears in an inner query string.

    Parameters
    ----------
    filename : str
        Local filename or URL.

    Returns
    -------
    str or None
        File suffix suitable for ``CACHE.build_filename()``, or ``None`` if
        no suffix can be determined.
    """
    def _extract(path: str) -> Optional[str]:
        suffixes = Path(path).suffixes
        if not suffixes:
            return None
        if suffixes[-1] == ".gz" and len(suffixes) > 1:
            return "".join(suffixes[-2:])
        return suffixes[-1]

    def _scan_query(query: str) -> Optional[str]:
        for values in urllib.parse.parse_qs(query).values():
            for v in values:
                leaf = v.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
                candidate = _extract(leaf)
                if candidate:
                    return candidate
                inner_query = urllib.parse.urlsplit(v).query
                if inner_query:
                    candidate = _scan_query(inner_query)
                    if candidate:
                        return candidate
        return None

    parsed = urllib.parse.urlsplit(filename)
    result = _extract(parsed.path)
    if result:
        return result

    return _scan_query(parsed.query)


@dataclass(frozen=True, repr=False)
class Decoder:
    """
    Decode cached resources using either their filesystem path or raw bytes.

    Parameters
    ----------
    func : Callable
        Function performing the actual decoding. Its first positional argument
        receives either a filename or bytes according to ``input_type``.
    input_type : {"FILE", "BYTES"}
        Required input representation for ``func``. ``FILE`` decoders receive
        a filesystem path; ``BYTES`` decoders receive raw bytes. Gzip-compressed
        formats (e.g. ``.nii.gz``, ``.csv.gz``) are handled natively by their
        respective readers when given a path.
    kwargs : dict
        Keyword arguments forwarded to ``func``.
    """

    func: Callable
    input_type: Literal["FILE", "BYTES"] = "BYTES"
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.input_type not in {"FILE", "BYTES"}:
            raise ValueError(
                f"Unknown decoder input type {self.input_type!r}. "
                "Expected 'FILE' or 'BYTES'."
            )

    def __repr__(self) -> str:
        func_name = getattr(self.func, "__qualname__", None) or getattr(self.func, "__name__", repr(self.func))
        module = getattr(self.func, "__module__", None)
        if module:
            func_name = f"{module}.{func_name}"
        kwargs_str = f", kwargs={self.kwargs!r}" if self.kwargs else ""
        return f"Decoder(func={func_name}, input_type={self.input_type!r}{kwargs_str})"

    @classmethod
    def from_filename(cls, filename: str) -> Optional["Decoder"]:
        """
        Find the decoder associated with a filename or URL.

        A trailing ``.gz`` is ignored when identifying the underlying format;
        gzip handling itself is delegated to the selected decoder.

        For backwards compatibility, an otherwise unknown ``*.gz`` resource
        is decoded into its uncompressed bytes.

        Parameters
        ----------
        filename : str
            Filename or URL whose suffix determines the decoder.

        Returns
        -------
        Decoder or None
            Matching decoder, or ``None`` if no decoder can be inferred.
        """
        suffix = _get_suffix(filename)
        if suffix is None:
            return None

        lookup = suffix[:-3] if suffix.endswith(".gz") else suffix

        if lookup in DECODERS:
            return DECODERS[lookup]

        return None

    def __call__(self, value: Union[str, bytes]):
        """
        Decode a value.

        Parameters
        ----------
        value : str or bytes
            Filesystem path for ``FILE`` decoders; raw bytes for ``BYTES``
            decoders.

        Returns
        -------
        object
            Result returned by the configured decoder function.
        """
        return self.func(value, **self.kwargs)


# Backwards-compatible entry point.
def find_suitable_decoder(filename: str, decoder_spec: dict = None) -> Optional[Decoder]:
    """
    Infer a decoder from a filename or URL.

    This function is retained for backwards compatibility. New code may use
    ``Decoder.from_filename()`` directly.
    """
    base = Decoder.from_filename(filename)
    if not decoder_spec:
        return base
    decoder_kwargs = {k: v for k, v in decoder_spec.items() if k != "@type"}
    return replace_dataclass_fields(base, kwargs=decoder_kwargs)


DECODERS = {
    ".nii": Decoder(
        load_nibabel,
        input_type="FILE",
    ),
    ".gii": Decoder(
        load_nibabel,
        input_type="FILE",
    ),
    ".json": Decoder(
        lambda b: json.loads(b.decode()),
    ),
    ".tck": Decoder(
        streamlines.load,
        input_type="FILE",
    ),
    ".csv": Decoder(
        pd.read_csv,
        input_type="FILE",
    ),
    ".tsv": Decoder(
        pd.read_csv,
        input_type="FILE",
        kwargs={"delimiter": "\t"},
    ),
    ".txt": Decoder(
        pd.read_csv,
        input_type="FILE",
        kwargs={
            "delimiter": " ",
            "header": None,
        },
    ),
    ".zip": Decoder(
        ZipFile,
        input_type="FILE",
    ),
    ".png": Decoder(
        skimage_io.imread,
        input_type="FILE",
    ),
    ".npy": Decoder(
        np.load,
        input_type="FILE",
    ),
    ".annot": Decoder(
        freesurfer.read_annot,
        input_type="FILE",
    ),
    ".h5": Decoder(
        h5py.File,
        input_type="FILE",
        kwargs={"mode": "r"},
    ),
    ".nwb": Decoder(
        h5py.File,
        input_type="FILE",
        kwargs={"mode": "r"},
    ),
}

if _HAS_TRX:
    DECODERS[".trx"] = Decoder(_load_trx, input_type="FILE")


class SiibraHttpRequestError(Exception):
    def __init__(self, url: str, status_code: int, msg="Cannot execute http request."):
        self.url = url
        self.status_code = status_code
        self.msg = msg
        Exception.__init__(self)

    def __str__(self):
        return f"{self.msg}\n\tStatus code: {self.status_code}\n\tUrl: {self.url:76.76}"


class HttpRequest:
    def __init__(
        self,
        url: str,
        func: Union[Decoder, Callable] = None,
        msg_if_not_cached: str = None,
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
        self._set_decoder_func(func)
        self.kwargs = kwargs
        self.cachefile = CACHE.build_filename(
            self.url + json.dumps(kwargs),
            suffix=_get_suffix(self.url)
        )
        self.msg_if_not_cached = msg_if_not_cached
        self.refresh = refresh
        self.post = post

    def _set_decoder_func(self, func: Callable = None):
        """
        Sets the decoder function of the HttpRequest. If `func` is None,
        it will try to find a suitable decoder.

        Parameters
        ----------
        func : Callable, default: None
        """
        if func is None:
            self.func = Decoder.from_filename(self.url)
        elif isinstance(func, Decoder):
            self.func = func
        else:
            # Existing user-provided func= callbacks continue receiving bytes.
            self.func = Decoder(
                func,
                input_type="BYTES",
            )

    @property
    def cached(self):
        return os.path.isfile(self.cachefile)

    def _retrieve(self, block_size=1024, min_bytesize_with_no_progress_info=2e8):
        """
        Populates the file cache with the data from http if required.
        noop if 1/ data is already cached and 2/ refresh flag not set
        The caller should load the cachefile after _retrieve successfully executes

        Safe to call concurrently: if several threads or processes request the
        same uncached file, it is downloaded only once.
        """
        if self.cached and not self.refresh:
            return

        def download(path):
            # Called by CACHE.fill under the cache lock, only if still needed.
            if self.msg_if_not_cached is not None:
                logger.info(self.msg_if_not_cached)

            headers = self.kwargs.get("headers", {})
            other_kwargs = {
                key: self.kwargs[key] for key in self.kwargs if key != "headers"
            }

            http_method = _SESSION.post if self.post else _SESSION.get
            try:
                r = http_method(
                    self.url,
                    headers={
                        **USER_AGENT_HEADER,
                        **headers,
                    },
                    **other_kwargs,
                    stream=True,
                )
            except requests.exceptions.ConnectionError as e:
                if not self.post and "RemoteDisconnected" in str(e):
                    logger.debug(f"Stale connection for {self.url}, retrying...")
                    adapter = _SESSION.get_adapter(url=self.url)
                    adapter.close()
                    r = http_method(self.url, headers={**USER_AGENT_HEADER, **headers}, **other_kwargs, stream=True)
                else:
                    raise e

            with r:  # release the connection, also when raising below
                if not r.ok:
                    raise SiibraHttpRequestError(status_code=r.status_code, url=self.url)

                size_bytes = int(r.headers.get("content-length", 0))
                show_progress = size_bytes > min_bytesize_with_no_progress_info
                if show_progress:
                    progress_bar = siibra_tqdm(
                        total=size_bytes,
                        unit="iB",
                        unit_scale=True,
                        position=0,
                        leave=True,
                        desc=f"Downloading {os.path.split(self.url)[-1]} ({size_bytes / 1024**2:.1f} MiB)",
                    )
                try:
                    with open(path, "wb") as f:
                        for data in r.iter_content(block_size):
                            if show_progress:
                                progress_bar.update(len(data))
                            f.write(data)
                finally:
                    if show_progress:
                        progress_bar.close()

        CACHE.fill(self.cachefile, download, replace=self.refresh)
        self.refresh = False

    def get(self):
        self._retrieve()
        try:
            if self.func is None:
                with open(self.cachefile, "rb") as f:
                    return f.read()

            if self.func.input_type == "FILE":
                return self.func(self.cachefile)
            else:
                with open(self.cachefile, "rb") as f:
                    return self.func(f.read())

        except Exception as e:
            # if network error results in bad cache, it may get raised here
            # e.g. BadZipFile("File is not a zip file")
            # if that happens, remove cachefile so the next call downloads it
            # again - but only if siibra wrote it: FileLoader points cachefile
            # at the user's own local file.
            if CACHE.contains(self.cachefile):
                try:
                    os.unlink(self.cachefile)
                except Exception:
                    pass
            raise e

    @property
    def data(self):
        # for backward compatibility with old LazyHttpRequest class
        return self.get()


class FileLoader(HttpRequest):
    """
    Just a loads a local file, but mimics the behaviour
    of cached http requests used in other connectors.
    """
    def __init__(self, filepath, func=None):
        HttpRequest.__init__(
            self,
            filepath,
            refresh=False,
            func=func,
        )
        p = Path(filepath)
        if not p.is_file() and not (p.is_dir() and p.suffix == ".trx"):
            raise ValueError(f"'{p}' is not a valid input, `FileLoader` takes a file or a '.trx'")
        self.cachefile = filepath

    def _retrieve(self, **kwargs):
        if kwargs:
            logger.warning(f"Keywords {list(kwargs.keys())} are supplied but can't be used for a local file.")


class ZipfileRequest(HttpRequest):
    def __init__(self, url, filename, func=None, refresh=False):
        HttpRequest.__init__(
            self, url, refresh=refresh,
            func=func or find_suitable_decoder(filename)
        )
        self.filename = filename
        if os.path.isfile(url):
            self.cachefile = url

    def get(self):
        self._retrieve()

        with ZipFile(self.cachefile) as zipfile:
            filenames = zipfile.namelist()
            matches = [
                fn for fn in filenames
                if fn == self.filename
            ]

            if len(matches) == 0:
                raise RuntimeError(
                    f"Requested filename {self.filename} not found "
                    f"in archive at {self.url}"
                )

            if len(matches) > 1:
                raise RuntimeError(
                    f"Requested filename {self.filename} was not unique "
                    f"in archive at {self.url}. Candidates were: "
                    f'{", ".join(matches)}'
                )

            member = matches[0]

            if self.func is None:
                with zipfile.open(member) as f:
                    return f.read()

            if self.func.input_type == "BYTES":
                with zipfile.open(member) as f:
                    return self.func(f.read())

            stat = os.stat(self.cachefile)

            member_cachefile = CACHE.build_filename(
                (
                    f"{self.url}:{member}:"
                    f"{stat.st_size}:{stat.st_mtime_ns}"
                ),
                suffix=_get_suffix(member),
            )

            def extract_member(tmp):
                with zipfile.open(member) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            CACHE.fill(member_cachefile, extract_member)
            return self.func(member_cachefile)


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
            logger.debug(
                f"token_endpoint exists in .well-known/openid-configuration. Setting _IAM_TOKEN_ENDPOINT to {json_resp.get('token_endpoint')}"
            )
            cls._IAM_TOKEN_ENDPOINT = json_resp.get("token_endpoint")
        else:
            logger.warning(
                "expect token endpoint in .well-known/openid-configuration, but was not present"
            )

        if "device_authorization_endpoint" in json_resp:
            logger.debug(
                f"device_authorization_endpoint exists in .well-known/openid-configuration. setting _IAM_DEVICE_ENDPOINT to {json_resp.get('device_authorization_endpoint')}"
            )
            cls._IAM_DEVICE_ENDPOINT = json_resp.get("device_authorization_endpoint")
        else:
            logger.warning(
                "expected device_authorization_endpoint in .well-known/openid-configuration, but was not present"
            )

    @classmethod
    def fetch_token(cls, **kwargs):
        """
        Fetch an EBRAINS token using commandline-supplied username/password
        using the data proxy endpoint.


        :ref:`Details on how to access EBRAINS are here.<accessEBRAINS>`
        """
        cls.device_flow(**kwargs)

    @classmethod
    def device_flow(cls, **kwargs):
        import sys

        if all(
            [
                not sys.__stdout__.isatty(),  # if is tty, do not raise
                not any(
                    k in ["JPY_INTERRUPT_EVENT", "JPY_PARENT_PID"] for k in os.environ
                ),  # if is notebook environment, do not raise
                not os.getenv(
                    "SIIBRA_ENABLE_DEVICE_FLOW"
                ),  # if explicitly enabled by env var, do not raise
            ]
        ):
            raise _exceptions.EbrainsAuthenticationError(
                "sys.stdout is not tty, SIIBRA_ENABLE_DEVICE_FLOW is not set,"
                "and not running in a notebook. Are you running in batch mode?"
            )

        cls.init_oidc()

        def get_scope() -> str:
            scope = kwargs.get("scope")
            if not scope:
                return None
            if not isinstance(scope, list):
                logger.warning("scope needs to be a list, is but is not... skipping")
                return None
            if not all(isinstance(scope, str) for scope in scope):
                logger.warning("scope needs to be all str, but is not")
                return None
            if len(scope) == 0:
                logger.warning("provided empty list as scope... skipping")
                return None
            return "+".join(scope)

        scope = get_scope()

        data = {"client_id": cls._IAM_DEVICE_FLOW_CLIENTID}

        if scope:
            data["scope"] = scope

        resp = requests.post(url=cls._IAM_DEVICE_ENDPOINT, data=data)
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
                message = (
                    f"exceeded max attempts: {cls._IAM_DEVICE_MAXTRIES}, aborting..."
                )
                logger.error(message)
                raise _exceptions.EbrainsAuthenticationError(message)
            attempt_number += 1
            resp = requests.post(
                url=cls._IAM_TOKEN_ENDPOINT,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": cls._IAM_DEVICE_FLOW_CLIENTID,
                    "device_code": device_code,
                },
            )

            if resp.status_code == 200:
                json_resp = resp.json()
                logger.debug("Device flow successful:", json_resp)
                cls._KG_API_TOKEN = json_resp.get("access_token")
                print("ebrains token successfully set.")
                break

            if resp.status_code == 400:
                json_resp = resp.json()
                error = json_resp.get("error")
                if error == "slow_down":
                    sleep_timer += 1
                logger.debug(f"400 error: {resp.content}")
                continue

            raise _exceptions.EbrainsAuthenticationError(resp.content)

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
                "\n 1. Let siibra get you a token by using siibra.fetch_ebrains_token() and follow the prompt."
                "\n 2. If you know how to get a token yourself, set it as $HBP_AUTH_TOKEN or siibra.set_ebrains_token()"
                "\n 3. If you are an application developer, you might configure keycloak access by setting $KEYCLOAK_CLIENT_ID"
                "and $KEYCLOAK_CLIENT_SECRET."
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


def try_all_connectors():
    def outer(fn):
        @wraps(fn)
        def inner(self: "GitlabProxyEnum", *args, **kwargs):
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
    DATASETVERSION_V3 = "DATASETVERSION_V3"

    @property
    def connectors(self) -> List["GitlabConnector"]:
        servers = [
            ("https://jugit.fz-juelich.de", 7846),
            ("https://gitlab.ebrains.eu", 421),
        ]
        from .repositories import GitlabConnector, LocalFileRepository

        if SIIBRA_USE_LOCAL_SNAPSPOT:
            logger.info(f"Using localsnapshot at {SIIBRA_USE_LOCAL_SNAPSPOT}")
            return [LocalFileRepository(SIIBRA_USE_LOCAL_SNAPSPOT)]
        else:
            return [
                GitlabConnector(server[0], server[1], "master", archive_mode=True)
                for server in servers
            ]

    @try_all_connectors()
    def search_files(
        self,
        folder: str,
        suffix=None,
        recursive=True,
        *,
        connector: "GitlabConnector" = None,
    ) -> List[str]:
        assert connector
        return connector.search_files(folder, suffix=suffix, recursive=recursive)

    @try_all_connectors()
    def get(self, filename, decode_func=None, *, connector: "GitlabConnector" = None):
        assert connector
        return connector.get(filename, "", decode_func)


class GitlabProxy(HttpRequest):
    folder_dict = {
        GitlabProxyEnum.DATASET_V1: "ebrainsquery/v1/dataset",
        GitlabProxyEnum.DATASET_V3: "ebrainsquery/v3/Dataset",
        GitlabProxyEnum.DATASETVERSION_V3: "ebrainsquery/v3/DatasetVersion",
        GitlabProxyEnum.PARCELLATIONREGION_V1: "ebrainsquery/v1/parcellationregions",
    }

    def __init__(
        self,
        flavour: GitlabProxyEnum,
        instance_id=None,
    ):
        if flavour not in GitlabProxyEnum:
            raise RuntimeError("Can only proxy enum members")

        self.flavour = flavour
        self.folder = self.folder_dict[flavour]
        self.instance_id = instance_id
        self.get = cache_user_fn(self.get)

    def get(self):
        if self.instance_id:
            return self.flavour.get(f"{self.folder}/{self.instance_id}.json")
        return {
            "results": self.flavour.get(f"{self.folder}/_all.json")
        }


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
            raise MultiSourceRequestException(
                "All requests failed:\n" + "\n".join(str(exc) for exc in exceptions)
            )

    @property
    def data(self):
        return self.get()
