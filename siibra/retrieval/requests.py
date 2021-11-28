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

import json
from zipfile import ZipFile
import requests
import os
from nibabel import Nifti1Image
import gzip

DECODERS = {
    ".nii.gz": lambda b: Nifti1Image.from_bytes(gzip.decompress(b)),
    ".nii": lambda b: Nifti1Image.from_bytes(b),
    ".json": lambda b: json.loads(b.decode()),
    ".txt": lambda b: b.decode(),
}


class HttpRequest:
    def __init__(
        self, url, func=None, status_code_messages={}, msg_if_not_cached=None, refresh=False, **kwargs
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
        status_code_messages : dict
            Optional dictionary of message strings to output in case of error,
            where keys are http status code.
        refresh : bool, default: False
            If True, a possibly cached content will be ignored and refreshed
        """
        assert url is not None
        self.url = url
        self.kwargs = kwargs
        self.status_code_messages = status_code_messages
        self.cachefile = CACHE.build_filename(self.url + json.dumps(kwargs))
        self.msg_if_not_cached = msg_if_not_cached
        self.refresh = refresh
        self._set_decoder_func(func, url)

    def _set_decoder_func(self, func, fileurl: str):
        if func is None:
            suitable_decoders = [dec for sfx, dec in DECODERS.items() if fileurl.endswith(sfx)]
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
            r = requests.get(self.url, **self.kwargs)
            if r.ok:
                with open(self.cachefile, "wb") as f:
                    f.write(r.content)
                self.refresh = False
                return r.content
            elif r.status_code in self.status_code_messages:
                raise RuntimeError(self.status_code_messages[r.status_code])
            else:
                print(self.kwargs)
                raise RuntimeError(
                    f"Could not retrieve data.\nhttp status code: {r.status_code}\nURL: {self.url}"
                )

    def get(self):
        data = self._retrieve()
        if data is None:
            with open(self.cachefile, "rb") as f:
                data = f.read()
        return data if self.func is None else self.func(data)

    @property
    def data(self):
        # for backward compatibility with old LazyHttpRequest class
        return self.get()

class ZipfileRequest(HttpRequest):
    def __init__(self, url, filename, func=None):
        HttpRequest.__init__(self, url, func=func )
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

    SC_MESSAGES = {
        401: "The provided EBRAINS authentication token is not valid",
        403: "No permission to access the given query",
        404: "Query with this id not found",
    }

    _KG_API_TOKEN = None

    server = "https://kg.humanbrainproject.eu"
    org = "minds"
    domain = "core"
    version = "v1.0.0"

    keycloak_endpoint = "https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect/token"

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
        self.params = params
        # NOTE: we do not pass params and header, here,
        # since we want to evaluate them late in the get() method.
        # This is nice because it allows to set env. variable KG_TOKEN only when
        # really needed, and not necessarily on package initialization.
        HttpRequest.__init__(
            self,
            url,
            DECODERS[".json"],
            self.SC_MESSAGES,
            msg_if_not_cached=f"Executing EBRAINS KG query {query_id}{inst_tail}",
        )

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
            self.__class__._KG_API_TOKEN = os.environ['HBP_AUTH_TOKEN']
            return self.__class__._KG_API_TOKEN

        # try KEYCLOAK. Requires the following environment variables set:
        # KEYCLOAK_ENDPOINT, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET  
        keycloak = {
            v:os.environ.get(f'KEYCLOAK_{v.upper()}') 
            for v in ['client_id','client_secret']
        }
        if None not in keycloak.values():
            logger.info("Getting an EBRAINS token via keycloak client configuration...")
            result = requests.post(
                self.__class__.keycloak_endpoint,
                data = (
                    f"grant_type=client_credentials&client_id={keycloak['client_id']}"
                    f"&client_secret={keycloak['client_secret']}"
                    "&scope=kg-nexus-role-mapping%20kg-nexus-service-account-mock"
                ),
                headers = {'content-type': 'application/x-www-form-urlencoded'}
            )
            try:
                content = json.loads(result.content.decode("utf-8"))
            except json.JSONDecodeError as error:
                logger.error(f"Invalid json from keycloak:{error}")
                self.__class__._KG_API_TOKEN = None
            if 'error' in content:
                logger.error(content['error_description'])
                self.__class__._KG_API_TOKEN = None
            self.__class__._KG_API_TOKEN = content['access_token']

        if self.__class__._KG_API_TOKEN is None:
            # No success getting the token
            raise RuntimeError(
                "No access token for EBRAINS Knowledge Graph found. "
                f"Please set $HBP_AUTH_TOKEN or use '{self.__class__.__name__}.set_token()', "
                "or configure keycloak access by setting $KEYCLOAK_ENDPOINT, $KEYCLOAK_CLIENT_ID "
                "and $KEYCLOAK_CLIENT_SECRET."
            )
        
        return self.__class__._KG_API_TOKEN

    def get(self):
        """Evaluate KG Token is evaluated only on executrion of the request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.kg_token}",
        }
        self.kwargs = {"headers": headers, "params": self.params}
        return super().get()
