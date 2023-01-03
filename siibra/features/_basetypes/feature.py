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

from .. import _anchor

from ..._commons import logger
from ...core import _concept
from ...core import space, region, parcellation

from typing import Union
from tqdm import tqdm


class Feature:
    """
    Base class for anatomically anchored data features.
    """

    VISIBLE_SUBCLASSES = {}

    def __init__(
        self,
        modality: str,
        description: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        """
        Parameters
        ----------
        modality: str
            A textual description of the type of measured information
        description: str
            A textual description of the feature.
        anchor: AnatomicalAnchor
        datasets : list
            list of datasets corresponding to this feature
        """
        self._modality_cached = modality
        self._description = description
        self._anchor_cached = anchor
        self.datasets = datasets

    @property
    def modality(self):
        # allows subclasses to implement lazy loading of an anchor
        return self._modality_cached

    @property
    def anchor(self):
        # allows subclasses to implement lazy loading of an anchor
        return self._anchor_cached

    @classmethod
    def _get_visible_subclass_names(cls):
        """ Returns the name of all visible subclasses. """
        return [
            c.__name__ for c in cls.VISIBLE_SUBCLASSES.values()
            if issubclass(c, cls)
        ]

    def __init_subclass__(cls, configuration_folder=None):
        if not cls.__name__.startswith('_'):
            assert cls.__name__ not in cls.VISIBLE_SUBCLASSES
            cls.VISIBLE_SUBCLASSES[cls.__name__] = cls
        cls._live_queries = []
        cls._preconfigured_instances = None
        cls._configuration_folder = configuration_folder
        return super().__init_subclass__()

    @property
    def description(self):
        """ Allowssubclasses to overwrite the description with a function call. """
        return self._description

    @property
    def name(self):
        """Returns a short human-readable name of this feature."""
        return f"{self.__class__.__name__} ({self.modality}) anchored at {self.anchor}"

    @classmethod
    def get_instances(cls, **kwargs):
        """
        Retrieve objects of a particular feature subclass.
        Objects can be preconfigured in the configuration,
        or delivered by Live queries.
        """
        if cls._preconfigured_instances is None:
            if cls._configuration_folder is None:
                cls._preconfigured_instances = []
            else:
                from ...configuration.configuration import Configuration
                conf = Configuration()
                Configuration.register_cleanup(cls.clean_instances)
                assert cls._configuration_folder in conf.folders
                cls._preconfigured_instances = [
                    o for o in conf.build_objects(cls._configuration_folder)
                    if isinstance(o, cls)
                ]
                logger.debug(
                    f"Built {len(cls._preconfigured_instances)} preconfigured {cls.__name__} "
                    f"objects from {cls._configuration_folder}."
                )

        return cls._preconfigured_instances

    @classmethod
    def clean_instances(cls):
        """ Removes all instantiated object instances"""
        cls._preconfigured_instances = None

    def matches(self, concept: _concept.AtlasConcept) -> bool:
        if self.anchor and self.anchor.matches(concept):
            self.anchor._last_matched_concept = concept
            return True
        self.anchor._last_matched_concept = None
        return False

    @property
    def last_match_result(self):
        return None if self.anchor is None \
            else self.anchor.last_match_result

    @property
    def last_match_description(self):
        return "" if self.anchor is None \
            else self.anchor.last_match_description

    @classmethod
    def match(cls, concept: Union[region.Region, parcellation.Parcellation, space.Space], feature_type: Union[str, type, list], **kwargs):
        """
        Retrieve data features of the desired modality.

        Parameters
        ----------
        concept: AtlasConcept
            An anatomical concept, typically a brain region or parcellation.
        modality: subclass of Feature
            specififies the type of features ("modality")
        """
        if isinstance(feature_type, list):
            assert all(isinstance(t, (str, cls)) for t in feature_type)
            return sum((cls.match(concept, t) for t in feature_type), [])
        elif isinstance(feature_type, str):
            feature_type = cls.VISIBLE_SUBCLASSES[feature_type]

        if not isinstance(concept, (region.Region, parcellation.Parcellation, space.Space)):
            raise ValueError(
                "Feature.match / siibra.get_features only accepts Region, "
                "Space and Parcellation objects as concept."
            )

        msg = f"Matching {feature_type.__name__} to {concept}"
        instances = feature_type.get_instances()
        preconfigured_instances = [
            f for f in tqdm(instances, desc=msg, total=len(instances))
            if f.matches(concept)
        ]

        live_instances = []
        for QueryType in feature_type._live_queries:
            argstr = f" ({', '.join('='.join(map(str,_)) for _ in kwargs.items())})" \
                if len(kwargs) > 0 else ""
            logger.info(
                f"Running live query for {QueryType.feature_type.__name__} "
                f"objects linked to {str(concept)}{argstr}"
            )
            q = QueryType(**kwargs)
            live_instances.extend(q.query(concept))

        return preconfigured_instances + live_instances
