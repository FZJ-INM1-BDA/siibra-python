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
from ..retrieval.repositories import GitlabConnector, RepositoryConnector
from ..retrieval.exceptions import NoSiibraConfigMirrorsAvailableException
from ..retrieval.requests import SiibraHttpRequestError

from typing import Union
from collections import defaultdict
from requests.exceptions import ConnectionError
from os import path


class Configuration:
    """
    Provides access to siibra configurations stored in different places,
    and on request builds preconfigured objects from selected subfolders
    of the the configuration.

    Configuration repositories and detected preconfiguration spec files
    are stored globally as class variables and shared by Configuration instances.
    """

    CONFIG_REPOS = [
        ("https://jugit.fz-juelich.de", 3484),
        ("https://gitlab.ebrains.eu", 93),
    ]

    CONFIGURATIONS = [
        GitlabConnector(server, project, "siibra-{}".format(__version__), skip_branchtest=True)
        for server, project in CONFIG_REPOS
    ]

    CONFIGURATION_EXTENSIONS = []

    _cleanup_funcs = []

    @staticmethod
    def get_folders(connector: RepositoryConnector):
        return {
            path.dirname(f)
            for f in connector.search_files(suffix='.json', recursive=True)
        }

    def __init__(self):

        # lists of loaders for json specification files
        # found in the siibra configuration, stored per
        # preconfigured class name. These files can
        # loaded and fed to the Factory.from_json
        # to produce the corresponding object.
        self.spec_loaders = defaultdict(list)

        # retrieve json spec loaders from the default configuration
        for connector in self.CONFIGURATIONS:
            try:
                for folder in self.get_folders(connector):
                    loaders = connector.get_loaders(folder, suffix='.json')
                    if len(loaders) > 0:
                        self.spec_loaders[folder] = loaders
                break
            except (ConnectionError, SiibraHttpRequestError):
                logger.error(f"Cannot load configuration from {str(connector)}")
                *_, last = self.CONFIGURATIONS
                if connector is last:
                    raise NoSiibraConfigMirrorsAvailableException(
                        "Tried all mirrors, none available."
                    )
        else:
            raise RuntimeError("Cannot pull any default siibra configuration.")

        # add additional spec loaders from extension configurations
        for connector in self.CONFIGURATION_EXTENSIONS:
            try:
                for folder in self.get_folders(connector):
                    self.spec_loaders[folder].extend(
                        connector.get_loaders(folder, suffix='json')
                    )
                break
            except ConnectionError:
                logger.error(f"Cannot connect to configuration extension {str(connector)}")
                continue

        logger.debug(
            "Preconfigurations: "
            + " | ".join(f"{folder}: {len(L)}" for folder, L in self.spec_loaders.items())
        )

    @property
    def folders(self):
        return list(self.spec_loaders.keys())

    @classmethod
    def use_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("Configuration needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Using custom configuration from {str(conn)}")
        cls.CONFIGURATIONS = [conn]
        # call registered cleanup functions
        for func in cls._cleanup_funcs:
            func()

    @classmethod
    def extend_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        if conn in cls.CONFIGURATION_EXTENSIONS:
            logger.warning(f"The configuration {str(conn)} is already registered.")
        else:
            logger.info(f"Extending configuration with {str(conn)}")
            cls.CONFIGURATION_EXTENSIONS.append(conn)
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

    def build_objects(self, folder: str, **kwargs):
        """
        Build the preconfigured objects of the specified class, if any.
        """
        result = []

        if folder not in self.folders:
            logger.warning(f"No configuration found for building from configuration folder {folder}.")
            return result

        from .factory import Factory
        specloaders = self.spec_loaders.get(folder, [])
        if len(specloaders) == 0:  # no loaders found in this configuration folder!
            return result

        obj0 = Factory.from_json(
            dict(
                specloaders[0][1].data,
                **{'filename': specloaders[0][0]}
            )
        )

        for fname, loader in siibra_tqdm(
            specloaders,
            total=len(specloaders),
            desc=f"Loading preconfigured {obj0.__class__.__name__} instances"
        ):
            # filename is added to allow Factory creating reasonable default object identifiers\
            obj = Factory.from_json(dict(loader.data, **{'filename': fname}))
            result.append(obj)

        return result

    def __getitem__(self, folder: str):
        return self.build_objects(folder)


if SIIBRA_USE_CONFIGURATION:
    logger.warning(f"config.SIIBRA_USE_CONFIGURATION defined, use configuration at {SIIBRA_USE_CONFIGURATION}")
    Configuration.use_configuration(SIIBRA_USE_CONFIGURATION)
