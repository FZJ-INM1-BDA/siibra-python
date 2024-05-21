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

from ..commons import logger, __version__, SIIBRA_USE_CONFIGURATION, siibra_tqdm
from ..retrieval.repositories import GitlabConnector, RepositoryConnector, GithubConnector
from ..retrieval.exceptions import NoSiibraConfigMirrorsAvailableException
from ..retrieval.requests import SiibraHttpRequestError

from typing import Union, List
from collections import defaultdict
from requests.exceptions import ConnectionError
from json import JSONDecodeError


class Configuration:
    """
    Provides access to siibra configuration specs stored in different places.
    On request builds preconfigured objects for specific spec types.

    Configuration repositories and detected preconfiguration spec files
    are stored globally as class variables and shared by Configuration instances.
    """

    CONFIG_REPOS = [
        (GithubConnector, "FZJ-INM1-BDA", "siibra-configurations"),
        (GitlabConnector, "https://gitlab.ebrains.eu", 892)
    ]
    CONFIG_CONNECTORS: List[RepositoryConnector] = [
        conn(
            server_or_owner,
            project_or_repo,
            reftag="siibra-{}".format(__version__),
            skip_branchtest=True
        )
        for conn, server_or_owner, project_or_repo in CONFIG_REPOS
    ]

    CONFIG_EXTENSIONS = []

    _cleanup_funcs = []

    def _load_specs(self, connector):
        """Adds valid configuration specs from a connector. """
        for fname in siibra_tqdm(
            connector.search_files(suffix='.json', recursive=True),
            desc=f"Loading siibra configuration files from {connector}"
        ):
            try:
                spec = connector.get(fname)
                if isinstance(spec, dict) and ("@type" in spec):
                    self.specs[spec.pop('@type')].append((fname, spec))
                else:
                    logger.debug(f"Skipped unknown specification in '{fname}'")
            except TypeError as e:
                print(str(e))
                logger.warning(f"Skipped invalid specification in '{fname}'")
            except JSONDecodeError as e:
                print(str(e))
                logger.warning(f"Skipped invalid json file '{fname}'")

    def __init__(self):

        # lists of configuration specs by configuration schema
        self.specs = defaultdict(list)

        # Retrieve configuration specs from the default configuration.
        # If this fails we stop.
        for connector in self.CONFIG_CONNECTORS:
            try:
                self._load_specs(connector)
                break  # controlled successful stop of the loop, needed to use "else:" below
            except (ConnectionError, SiibraHttpRequestError):
                logger.error(f"Cannot load configuration from {str(connector)}")
                *_, last = self.CONFIG_CONNECTORS
                if connector is last:
                    raise NoSiibraConfigMirrorsAvailableException(
                        "Tried all mirrors, none available."
                    )
        else:
            raise RuntimeError("Cannot load any default configuration.")

        # Add  configuration specs from optional extension configurations.
        # If this fails, we continue.
        for connector in self.CONFIG_EXTENSIONS:
            try:
                self._load_specs(connector)
                break
            except ConnectionError:
                logger.error(f"Cannot load configuration extension {str(connector)}")
                continue

        logger.debug(
            "Preconfigurations: "
            + " | ".join(f"{schema}: {len(L)}" for schema, L in self.specs.items())
        )

    @property
    def known_schemas(self):
        return list(self.specs.keys())

    @classmethod
    def use_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("Configuration needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Using custom configuration from {str(conn)}")
        cls.CONFIG_CONNECTORS = [conn]
        # call registered cleanup functions
        for func in cls._cleanup_funcs:
            func()

    @classmethod
    def extend_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        if conn in cls.CONFIG_EXTENSIONS:
            logger.warning(f"The configuration {str(conn)} is already registered.")
        else:
            logger.info(f"Extending configuration with {str(conn)}")
            cls.CONFIG_EXTENSIONS.append(conn)
            # call registered cleanup functions
            for func in cls._cleanup_funcs:
                func()

    @classmethod
    def register_cleanup(cls, func):
        """
        Register an arbitrary function that should be executed when the
        configuration is changed, e.g. with use_configuration().
        """
        cls._cleanup_funcs.append(func)

    def build_objects(self, schema: str, **kwargs):
        """
        Build preconfigured objects matching the given configuration schema.
        """
        result = []

        if schema not in self.specs:
            logger.warning(f"No configuration found for building from configuration type {schema}.")
            return result

        from .factory import Factory
        specs = self.specs.get(schema, [])
        if len(specs) == 0:  # no specs found in this configuration schema
            return result

        for fname, spec in siibra_tqdm(
            specs, total=len(specs),
            desc=f"Loading preconfigured instances for {schema}"
        ):
            # filename is added to allow Factory creating reasonable default object identifiers\
            obj = Factory.from_json(dict(spec, **{'filename': fname, '@type': schema}))
            result.extend(obj) if isinstance(obj, list) else result.append(obj)

        return result

    def __getitem__(self, schema: str):
        return self.build_objects(schema)


if SIIBRA_USE_CONFIGURATION:
    logger.warning(f"config.SIIBRA_USE_CONFIGURATION defined, use configuration at {SIIBRA_USE_CONFIGURATION}")
    Configuration.use_configuration(SIIBRA_USE_CONFIGURATION)
