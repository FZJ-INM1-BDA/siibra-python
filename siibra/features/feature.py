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
"""Handles multimodal data features and related queries."""

from . import anchor as _anchor

from ..commons import logger, InstanceTable, siibra_tqdm, __version__
from ..core import concept
from ..core import space, region, parcellation

from typing import Union, TYPE_CHECKING, List, Dict, Type, Tuple, BinaryIO
from hashlib import md5
from collections import defaultdict
from zipfile import ZipFile

if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsBaseDataset
    TypeDataset = EbrainsBaseDataset


class ParseLiveQueryIdException(Exception):
    pass


class EncodeLiveQueryIdException(Exception):
    pass


class NotFoundException(Exception):
    pass


_README_TMPL = """
Downloaded from siibra toolsuite.
siibra-python version: {version}

All releated resources (e.g. doi, web resources) are categorized under publications.

Name
----
{name}

Description
-----------
{description}

Modality
--------
{modality}

{publications}
"""
_README_PUBLICATIONS = """
Publications
------------
{doi}

{ebrains_page}

{authors}

{publication_desc}

"""


class Feature:
    """
    Base class for anatomically anchored data features.
    """

    SUBCLASSES: Dict[Type['Feature'], List[Type['Feature']]] = defaultdict(list)

    CATEGORIZED: Dict[str, Type['InstanceTable']] = defaultdict(InstanceTable)

    category: str = None

    def __init__(
        self,
        modality: str,
        description: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: List['TypeDataset'] = [],
        prerelease: bool = False,
        id: str = None,

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
        self._prerelease = prerelease
        self._id = id

    @property
    def modality(self):
        # allows subclasses to implement lazy loading of the modality
        return self._modality_cached

    @property
    def anchor(self):
        # allows subclasses to implement lazy loading of an anchor
        return self._anchor_cached

    def __init_subclass__(cls, configuration_folder=None, category=None, do_not_index=False):

        # Feature.SUBCLASSES serves as an index where feature class inheritance is cached. When users
        # queries a branch on the hierarchy, all children will also be queried. There are usecases where
        # such behavior is not desired (e.g. ProxyFeature, which wraps livequery features id to capture the
        # query context).
        # do_not_index flag allow the default index behavior to be toggled off.

        if do_not_index is False:

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
        cls.category = category
        if category is not None:
            cls.CATEGORIZED[category].add(cls.__name__, cls)
        return super().__init_subclass__()

    @classmethod
    def _get_subclasses(cls):
        return {Cls.__name__: Cls for Cls in cls.SUBCLASSES}

    @property
    def description(self):
        """Allows subclasses to overwrite the description with a function call."""
        if self._description:
            return self._description
        for ds in self.datasets:
            if ds.description:
                return ds.description
        return ''

    @property
    def LICENSE(self) -> str:
        licenses = []
        for ds in self.datasets:
            if ds.LICENSE is None or ds.LICENSE == "No license information is found.":
                continue
            if isinstance(ds.LICENSE, str):
                licenses.append(ds.LICENSE)
            if isinstance(ds.LICENSE, list):
                licenses.extend(ds.LICENSE)
        return '\n'.join(licenses)

    @property
    def doi_or_url(self) -> str:
        return '\n'.join([
            url.get("url")
            for ds in self.datasets
            for url in ds.urls
        ])

    @property
    def authors(self):
        return [
            contributer['name']
            for ds in self.datasets
            for contributer in ds.contributors
        ]

    @property
    def name(self):
        """Returns a short human-readable name of this feature."""
        name_ = f"{self.__class__.__name__} ({self.modality}) anchored at {self.anchor}"
        return name_ if not self._prerelease else f"[PRERELEASE] {name_}"

    @classmethod
    def get_instances(cls, **kwargs) -> List['Feature']:
        """
        Retrieve objects of a particular feature subclass.
        Objects can be preconfigured in the configuration,
        or delivered by Live queries.
        """
        if not hasattr(cls, "_preconfigured_instances"):
            return []

        if cls._preconfigured_instances is not None:
            return cls._preconfigured_instances

        if cls._configuration_folder is None:
            cls._preconfigured_instances = []
            return cls._preconfigured_instances

        from ..configuration.configuration import Configuration
        conf = Configuration()
        Configuration.register_cleanup(cls.clean_instances)
        if cls._configuration_folder not in conf.folders:
            logger.debug(f"{cls._configuration_folder} is not in current configuration")
            return []

        cls._preconfigured_instances = [
            o for o in conf.build_objects(cls._configuration_folder)
            if isinstance(o, cls)
        ]
        logger.debug(
            f"Built {len(cls._preconfigured_instances)} preconfigured {cls.__name__} "
            f"objects from {cls._configuration_folder}."
        )
        return cls._preconfigured_instances

    def plot(self, *args, **kwargs):
        """Feature subclasses override this with their customized plot methods."""
        raise NotImplementedError("Generic feature class does not have a standardized plot.")

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
        if self._id:
            return self._id

        prefix = ''
        for ds in self.datasets:
            if hasattr(ds, "id"):
                prefix = ds.id + '--'
                break
        name_ = self.name.lstrip("[PRERELEASE] ")
        return prefix + md5(name_.encode("utf-8")).hexdigest()

    def _export(self, fh: ZipFile):
        """
        Internal implementation. Subclasses can override but call super()._export(fh).
        This allows all classes in the __mro__ to have the opportunity to append files
        of interest.
        """
        ebrains_page = "\n".join(
            {ds.ebrains_page for ds in self.datasets if getattr(ds, "ebrains_page", None)}
        )
        doi = "\n".join({
            u.get("url")
            for ds in self.datasets if ds.urls
            for u in ds.urls
        })
        authors = ", ".join({
            cont.get('name')
            for ds in self.datasets if ds.contributors
            for cont in ds.contributors
        })
        publication_desc = "\n".join({ds.description for ds in self.datasets})
        if (ebrains_page or doi) and authors:
            publications = _README_PUBLICATIONS.format(
                ebrains_page="EBRAINS page\n" + ebrains_page if ebrains_page else "",
                doi="DOI\n" + doi if doi else "",
                authors="Authors\n" + authors if authors else "",
                publication_desc="Publication description\n" + publication_desc if publication_desc else ""
            )
        else:
            publications = "Note: could not obtain any publication information. The data may not have been published yet."
        fh.writestr(
            "README.md",
            _README_TMPL.format(
                version=__version__,
                name=self.name,
                description=self.description,
                modality=self.modality,
                publications=publications
            )
        )

    def export(self, filelike: Union[str, BinaryIO]):
        """
        Export as a zip archive.

        Args:
            filelike (string or filelike): name or filehandle to write the zip file. User is responsible to ensure the correct extension (.zip) is set.
        """
        fh = ZipFile(filelike, "w")
        self._export(fh)
        fh.close()

    @staticmethod
    def serialize_query_context(feat: 'Feature', concept: concept.AtlasConcept) -> str:
        """
        Serialize feature from livequery and query context.

        It is currently impossible to retrieve a livequery with a generic UUID.
        As such, the query context (e.g. region, space or parcellation) needs to
        be encoded in the id.

        Whilst it is possible to (de)serialize *any* queries, the method is setup to only serialize
        livequery features.

        The serialized livequery id follows the following pattern:

        <livequeryid_version>::<feature_cls_name>::<query_context>::<unserialized_id>

        Where:

        - livequeryid_version: version of the serialization. (e.g. lq0)
        - feature_cls_name: class name to query. (e.g. BigBrainIntensityProfile)
        - query_context: string to retrieve atlas concept in the query context. Can be one of the following:
            - s:<space_id>
            - p:<parcellation_id>
            - p:<parcellation_id>::r:<region_id>
        - unserialized_id: id prior to serialization

        See test/features/test_feature.py for tests and usages.
        """
        if not hasattr(feat.__class__, '_live_queries'):
            raise EncodeLiveQueryIdException(f"generate_livequery_featureid can only be used on live queries, but {feat.__class__.__name__} is not.")

        encoded_c = []
        if isinstance(concept, space.Space):
            encoded_c.append(f"s:{concept.id}")
        elif isinstance(concept, parcellation.Parcellation):
            encoded_c.append(f"p:{concept.id}")
        elif isinstance(concept, region.Region):
            encoded_c.append(f"p:{concept.parcellation.id}")
            encoded_c.append(f"r:{concept.name}")

        if len(encoded_c) == 0:
            raise EncodeLiveQueryIdException("no concept is encoded")

        return f"lq0::{feat.__class__.__name__}::{'::'.join(encoded_c)}::{feat.id}"

    @classmethod
    def deserialize_query_context(Cls, feature_id: str) -> Tuple[Type['Feature'], concept.AtlasConcept, str]:
        """
        Deserialize id into query context.

        See docstring of serialize_query_context for context.
        """
        lq_version, *rest = feature_id.split("::")
        if lq_version != "lq0":
            raise ParseLiveQueryIdException("livequery id must start with lq0::")

        clsname, *concepts, fid = rest

        Features = Cls.parse_featuretype(clsname)

        if len(Features) == 0:
            raise ParseLiveQueryIdException(f"classname {clsname!r} could not be parsed correctly. {feature_id!r}")
        F = Features[0]

        concept = None
        for c in concepts:
            if c.startswith("s:"):
                if concept is not None:
                    raise ParseLiveQueryIdException("Conflicting spec.")
                concept = space.Space.registry()[c.replace("s:", "")]
            if c.startswith("p:"):
                if concept is not None:
                    raise ParseLiveQueryIdException("Conflicting spec.")
                concept = parcellation.Parcellation.registry()[c.replace("p:", "")]
            if c.startswith("r:"):
                if concept is None:
                    raise ParseLiveQueryIdException("region has been encoded, but parcellation has not been populated in the encoding, {feature_id!r}")
                if not isinstance(concept, parcellation.Parcellation):
                    raise ParseLiveQueryIdException("region has been encoded, but previous encoded concept is not parcellation")
                concept = concept.get_region(c.replace("r:", ""))
        if concept is None:
            raise ParseLiveQueryIdException(f"concept was not populated: {feature_id!r}")

        return (F, concept, fid)

    @classmethod
    def parse_featuretype(cls, feature_type: str) -> List[Type['Feature']]:
        ftypes = {
            feattype
            for FeatCls, feattypes in cls.SUBCLASSES.items()
            if all(w.lower() in FeatCls.__name__.lower() for w in feature_type.split())
            for feattype in feattypes
        }
        if len(ftypes) > 1:
            return [ft for ft in ftypes if getattr(ft, 'category')]
        else:
            return list(ftypes)

    @classmethod
    def livequery(cls, concept: Union[region.Region, parcellation.Parcellation, space.Space], **kwargs) -> List['Feature']:
        if not hasattr(cls, "_live_queries"):
            return []

        live_instances = []
        for QueryType in cls._live_queries:
            argstr = f" ({', '.join('='.join(map(str,_)) for _ in kwargs.items())})" \
                if len(kwargs) > 0 else ""
            logger.info(
                f"Running live query for {QueryType.feature_type.__name__} "
                f"objects linked to {str(concept)}{argstr}"
            )
            q = QueryType(**kwargs)
            features = [
                Feature.wrap_livequery_feature(feat, Feature.serialize_query_context(feat, concept))
                for feat in q.query(concept)
            ]
            live_instances.extend(features)
        return live_instances

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
            assert all(
                (isinstance(t, str) or issubclass(t, cls))
                for t in feature_type
            )
            return list(dict.fromkeys(
                sum((
                    cls.match(concept, t, **kwargs) for t in feature_type
                ), [])
            ))

        if isinstance(feature_type, str):
            # feature type given as a string. Decode the corresponding class.
            # Some string inputs, such as connectivity, may hit multiple matches.
            ftype_candidates = cls.parse_featuretype(feature_type)
            if len(ftype_candidates) == 0:
                raise ValueError(
                    f"feature_type {str(feature_type)} did not match with any "
                    f"features. Available features are: {', '.join(cls.SUBCLASSES.keys())}"
                )
            logger.info(
                f"'{feature_type}' decoded as feature type/s: "
                f"{[c.__name__ for c in ftype_candidates]}."
            )
            return cls.match(concept, ftype_candidates, **kwargs)

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

        preconfigured_instances = [
            f for f in siibra_tqdm(
                instances, desc=msg, total=len(instances), disable=(not instances)
            ) if f.matches(concept)
        ]

        live_instances = feature_type.livequery(concept, **kwargs)

        return list(dict.fromkeys(preconfigured_instances + live_instances))

    @classmethod
    def get_instance_by_id(cls, feature_id: str, **kwargs):
        try:
            F, concept, fid = cls.deserialize_query_context(feature_id)
            return [
                f
                for f in F.livequery(concept, **kwargs)
                if f.id == fid or f.id == feature_id
            ][0]
        except ParseLiveQueryIdException:
            return [
                inst
                for Cls in Feature.SUBCLASSES[Feature]
                for inst in Cls.get_instances()
                if inst.id == feature_id
            ][0]
        except IndexError:
            raise NotFoundException

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

    @staticmethod
    def wrap_livequery_feature(feature: 'Feature', fid: str):
        """
        Wrap live query features, override only the id attribute.

        Some features do not have setters for the id property. The ProxyFeature class
        allow the id property to be overridden without touching the underlying class.

        See docstring of serialize_query_context for further context.
        """
        class ProxyFeature(feature.__class__, do_not_index=True):

            # override __class__ property
            # some instances of features accesses inst.__class__
            @property
            def __class__(self):
                return self.inst.__class__

            def __init__(self, inst: Feature, fid: str):
                self.inst = inst
                self.fid = fid

            def __str__(self) -> str:
                return self.inst.__str__()

            def __repr__(self) -> str:
                return self.inst.__repr__()

            @property
            def id(self):
                return self.fid

            def __getattr__(self, __name: str):
                return getattr(self.inst, __name)

        return ProxyFeature(feature, fid)
