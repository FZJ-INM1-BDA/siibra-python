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

from .. import anchor as _anchor

from ...commons import logger
from ...core import concept
from ...core import space, region, parcellation

from typing import Union, TYPE_CHECKING, List, Dict, Type
from tqdm import tqdm
from hashlib import md5
from collections import defaultdict

if TYPE_CHECKING:
    from ...retrieval.datasets import EbrainsDataset
    TypeDataset = EbrainsDataset


class Feature:
    """
    Base class for anatomically anchored data features.
    """

    SUBCLASSES: Dict[Type['Feature'], List[Type['Feature']]] = defaultdict(list)

    def __init__(
        self,
        modality: str,
        description: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: List['TypeDataset'] = []
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

    def __init_subclass__(cls, configuration_folder=None):
        # extend the subclass lists

        # Iterate over all mro, not just immediate base classes
        for BaseCls in cls.__mro__:
            # some base classes may not be sub class of feature, ignore these
            if not issubclass(BaseCls, Feature):
                continue
            cls.SUBCLASSES[BaseCls].append(cls)

        cls._live_queries = []
        cls._preconfigured_instances = None
        cls._configuration_folder = configuration_folder
        return super().__init_subclass__()

    @classmethod
    def _get_subclasses(cls):
        return {Cls.__name__: Cls for Cls in cls.SUBCLASSES}

    @property
    def description(self):
        """ Allowssubclasses to overwrite the description with a function call. """
        return self._description

    @property
    def name(self):
        """Returns a short human-readable name of this feature."""
        return f"{self.__class__.__name__} ({self.modality}) anchored at {self.anchor}"

    @classmethod
    def get_instances(cls, **kwargs) -> List['Feature']:
        """
        Retrieve objects of a particular feature subclass.
        Objects can be preconfigured in the configuration,
        or delivered by Live queries.
        """
        result = []

        if hasattr(cls, "_preconfigured_instances"):
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
            result.extend(cls._preconfigured_instances)

        return result

    @classmethod
    def clean_instances(cls):
        """ Removes all instantiated object instances"""
        cls._preconfigured_instances = None

    def matches(self, concept: concept.AtlasConcept) -> bool:
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

    @property
    def id(self):
        prefix = ''
        id_set = {ds.id for ds in self.datasets if hasattr(ds, 'id')}
        if len(id_set) == 1:
            prefix = list(id_set)[0] + '--'
        return prefix + md5(self.name.encode("utf-8")).hexdigest()

    @classmethod
    def match(cls, concept: Union[region.Region, parcellation.Parcellation, space.Space], feature_type: Union[str, Type['Feature'], list], **kwargs) -> List['Feature']:
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
            # a list of feature types is given, collect match results on those
            assert all((isinstance(t, str) or issubclass(t, cls)) for t in feature_type)
            return sum((cls.match(concept, t, **kwargs) for t in feature_type), [])

        if isinstance(feature_type, str):
            # feature type given as a string. Decode the corresponding class.
            # Some string inputs, such as connectivity, may hit multiple matches
            # In this case
            candidates = [
                feattype
                for FeatCls, feattypes in cls.SUBCLASSES.items()
                if all(w.lower() in FeatCls.__name__.lower() for w in feature_type.split())
                for feattype in feattypes
            ]
            if len(candidates) == 0:
                raise ValueError(f"feature_type {str(feature_type)} did not match with any features. Available features are: {', '.join(cls.SUBCLASSES.keys())}")

            return [feat for c in candidates for feat in cls.match(concept, c, **kwargs)]

        assert issubclass(feature_type, Feature)

        if not isinstance(concept, (region.Region, parcellation.Parcellation, space.Space)):
            raise ValueError(
                "Feature.match / siibra.features.get only accepts Region, "
                "Space and Parcellation objects as concept."
            )

        msg = f"Matching {feature_type.__name__} to {concept}"
        instances = [
            instance
            for f_type in cls.SUBCLASSES[feature_type]
            for instance in f_type.get_instances()
        ]

        if logger.getEffectiveLevel() > 20:
            preconfigured_instances = [f for f in instances if f.matches(concept)]
        else:
            preconfigured_instances = [f for f in tqdm(instances, desc=msg, total=len(instances)) if f.matches(concept)]

        live_instances = []
        if hasattr(feature_type, "_live_queries"):
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

    @classmethod
    def get_ascii_tree(cls):
        # build an Ascii representation of class hierarchy
        # under this feature class
        from anytree.importer import DictImporter
        from anytree import RenderTree

        def create_treenode(feature_type):
            return {
                'name': feature_type.__name__,
                'children': [
                    create_treenode(c) 
                    for c in feature_type.__subclasses__()
                ]
            }
        D = create_treenode(cls)
        importer = DictImporter()
        tree = importer.import_(D)
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node in RenderTree(tree)
        )
