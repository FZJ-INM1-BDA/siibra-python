# Copyright 2018-2023
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
from ..core import concept, space, region, parcellation, structure
from ..volumes import volume

from typing import Union, TYPE_CHECKING, List, Dict, Type, Tuple, BinaryIO, Any, Iterator
from hashlib import md5
from collections import defaultdict
from zipfile import ZipFile
from abc import ABC

if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsDataset
    TypeDataset = EbrainsDataset


class ParseLiveQueryIdException(Exception):
    pass


class EncodeLiveQueryIdException(Exception):
    pass


class NotFoundException(Exception):
    pass


class ParseCompoundFeatureIdException(Exception):
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

    _SUBCLASSES: Dict[Type['Feature'], List[Type['Feature']]] = defaultdict(list)
    _CATEGORIZED: Dict[str, Type['InstanceTable']] = defaultdict(InstanceTable)

    category: str = None

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
        # allows subclasses to implement lazy loading of the modality
        return self._modality_cached

    @property
    def anchor(self):
        # allows subclasses to implement lazy loading of an anchor
        return self._anchor_cached

    def __init_subclass__(cls, configuration_folder=None, category=None, do_not_index=False, **kwargs):

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
                cls._SUBCLASSES[BaseCls].append(cls)

        cls._live_queries = []
        cls._preconfigured_instances = None
        cls._configuration_folder = configuration_folder
        cls.category = category
        if category is not None:
            cls._CATEGORIZED[category].add(cls.__name__, cls)
        return super().__init_subclass__(**kwargs)

    @classmethod
    def _get_subclasses(cls):
        return {Cls.__name__: Cls for Cls in cls._SUBCLASSES}

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
        licenses = {ds.LICENSE for ds in self.datasets if ds.LICENSE}
        if not licenses:
            return "No license information is found."
        if len(licenses) == 1:
            return next(iter(licenses))
        logger.info("Found multiple licenses corresponding to datasets.")
        return '\n'.join(licenses)

    @property
    def urls(self) -> List[str]:
        """The list of URLs (including DOIs) associated with this feature."""
        return [
            url.get("url")
            for ds in self.datasets
            for url in ds.urls
        ]

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
        return f"{self.__class__.__name__} ({self.modality}) anchored at {self.anchor}"

    @classmethod
    def _get_instances(cls, **kwargs) -> List['Feature']:
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
        Configuration.register_cleanup(cls._clean_instances)
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

    def plot(self, *args, **kwargs):
        """Feature subclasses override this with their customized plot methods."""
        raise NotImplementedError("Generic feature class does not have a standardized plot.")

    @classmethod
    def _clean_instances(cls):
        """ Removes all instantiated object instances"""
        cls._preconfigured_instances = None

    def matches(self, concept: structure.BrainStructure, restrict_space: bool = False) -> bool:
        """
        Match the features anatomical anchor agains the given query concept.
        Record the most recently matched concept for inspection by the caller.
        """
        # TODO: storing the last matched concept. It is not ideal, might cause problems in multithreading
        if self.anchor and self.anchor.matches(concept, restrict_space):
            self.anchor._last_matched_concept = concept
            return True
        self.anchor._last_matched_concept = None
        return False

    @property
    def last_match_result(self):
        "The result of the last anchor comparison to a BrainStructure."
        return None if self.anchor is None else self.anchor.last_match_result

    @property
    def last_match_description(self):
        "The description of the last anchor comparison to a BrainStructure."
        return "" if self.anchor is None else self.anchor.last_match_description

    @property
    def id(self):
        prefix = ''
        for ds in self.datasets:
            if hasattr(ds, "id"):
                prefix = ds.id + '--'
                break
        return prefix + md5(self.name.encode("utf-8")).hexdigest()

    def _to_zip(self, fh: ZipFile):
        """
        Internal implementation. Subclasses can override but call super()._to_zip(fh).
        This allows all classes in the __mro__ to have the opportunity to append files
        of interest.
        """
        if isinstance(self, Compoundable) and "README.md" in fh.namelist():
            return
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

    def to_zip(self, filelike: Union[str, BinaryIO]):
        """
        Export as a zip archive.

        Parameters
        ----------
        filelike: str or path
            Filelike to write the zip file. User is responsible to ensure the
            correct extension (.zip) is set.
        """
        fh = ZipFile(filelike, "w")
        self._to_zip(fh)
        fh.close()

    @staticmethod
    def _serialize_query_context(feat: 'Feature', concept: concept.AtlasConcept) -> str:
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

        encoded_c = Feature._encode_concept(concept)

        return f"lq0::{feat.__class__.__name__}::{encoded_c}::{feat.id}"

    @classmethod
    def _deserialize_query_context(cls, feature_id: str) -> Tuple[Type['Feature'], concept.AtlasConcept, str]:
        """
        Deserialize id into query context.

        See docstring of serialize_query_context for context.
        """
        lq_version, *rest = feature_id.split("::")
        if lq_version != "lq0":
            raise ParseLiveQueryIdException("livequery id must start with lq0::")

        clsname, *concepts, fid = rest

        Features = cls._parse_featuretype(clsname)

        if len(Features) == 0:
            raise ParseLiveQueryIdException(f"classname {clsname!r} could not be parsed correctly. {feature_id!r}")
        F = Features[0]

        concept = cls._decode_concept(concepts)

        return (F, concept, fid)

    @staticmethod
    def _encode_concept(concept: concept.AtlasConcept):
        encoded_c = []
        if isinstance(concept, space.Space):
            encoded_c.append(f"s:{concept.id}")
        elif isinstance(concept, parcellation.Parcellation):
            encoded_c.append(f"p:{concept.id}")
        elif isinstance(concept, region.Region):
            encoded_c.append(f"p:{concept.parcellation.id}")
            encoded_c.append(f"r:{concept.name}")
        elif isinstance(concept, volume.Volume):
            encoded_c.append(f"v:{concept.name}")

        if len(encoded_c) == 0:
            raise EncodeLiveQueryIdException("no concept is encoded")

        return '::'.join(encoded_c)

    @classmethod
    def _decode_concept(cls, concepts: List[str]) -> concept.AtlasConcept:
        # choose exception to divert try-except correctly
        if issubclass(cls, CompoundFeature):
            exception = ParseCompoundFeatureIdException
        else:
            exception = ParseLiveQueryIdException

        concept = None
        for c in concepts:
            if c.startswith("s:"):
                if concept is not None:
                    raise exception("Conflicting spec.")
                concept = space.Space.registry()[c.replace("s:", "")]
            if c.startswith("p:"):
                if concept is not None:
                    raise exception("Conflicting spec.")
                concept = parcellation.Parcellation.registry()[c.replace("p:", "")]
            if c.startswith("r:"):
                if concept is None:
                    raise exception("region has been encoded, but parcellation has not been populated in the encoding, {feature_id!r}")
                if not isinstance(concept, parcellation.Parcellation):
                    raise exception("region has been encoded, but previous encoded concept is not parcellation")
                concept = concept.get_region(c.replace("r:", ""))

        if concept is None:
            raise ParseLiveQueryIdException("concept was not populated in feature id")
        return concept

    @classmethod
    def _parse_featuretype(cls, feature_type: str) -> List[Type['Feature']]:
        ftypes = sorted({
            feattype
            for FeatCls, feattypes in cls._SUBCLASSES.items()
            if all(w.lower() in FeatCls.__name__.lower() for w in feature_type.split())
            for feattype in feattypes
        }, key=lambda t: t.__name__)
        if len(ftypes) > 1:
            return [ft for ft in ftypes if getattr(ft, 'category')]
        else:
            return list(ftypes)

    @classmethod
    def _livequery(cls, concept: Union[region.Region, parcellation.Parcellation, space.Space], **kwargs) -> List['Feature']:
        if not hasattr(cls, "_live_queries"):
            return []

        live_instances = []
        for QueryType in cls._live_queries:
            argstr = f" ({', '.join('='.join(map(str,_)) for _ in kwargs.items())})" \
                if len(kwargs) > 0 else ""
            logger.debug(
                f"Running live query for {QueryType.feature_type.__name__} "
                f"objects linked to {str(concept)}{argstr}"
            )
            q = QueryType(**kwargs)
            features = q.query(concept)
            live_instances.extend(
                Feature._wrap_livequery_feature(f, Feature._serialize_query_context(f, concept))
                for f in features
            )

        return live_instances

    @classmethod
    def _match(
        cls,
        concept: structure.BrainStructure,
        feature_type: Union[str, Type['Feature'], list],
        restrict_space: bool = False,
        **kwargs
    ) -> List['Feature']:
        """
        Retrieve data features of the requested feature type (i.e. modality).
        This will
        - call Feature.match(concept) for any registered preconfigured features
        - run any registered live queries
        The preconfigured and live query instances are merged and returend as a list.

        If multiple feature types are given, recurse for each of them.


        Parameters
        ----------
        concept: AtlasConcept
            An anatomical concept, typically a brain region or parcellation.
        feature_type: subclass of Feature, str
            specififies the type of features ("modality")
        restrict_space: bool: default: False
            If true, will skip features anchored at spatial locations of
            different spaces than the concept. Requires concept to be a
            Location.
        """
        if isinstance(feature_type, list):
            # a list of feature types is given, collect match results on those
            assert all(
                (isinstance(t, str) or issubclass(t, cls))
                for t in feature_type
            )
            return list(dict.fromkeys(
                sum((
                    cls._match(concept, t, restrict_space, **kwargs) for t in feature_type
                ), [])
            ))

        if isinstance(feature_type, str):
            # feature type given as a string. Decode the corresponding class.
            # Some string inputs, such as connectivity, may hit multiple matches.
            ftype_candidates = cls._parse_featuretype(feature_type)
            if len(ftype_candidates) == 0:
                raise ValueError(
                    f"feature_type {str(feature_type)} did not match with any "
                    f"features. Available features are: {', '.join(cls._SUBCLASSES.keys())}"
                )
            logger.info(
                f"'{feature_type}' decoded as feature type/s: "
                f"{[c.__name__ for c in ftype_candidates]}."
            )
            return cls._match(concept, ftype_candidates, restrict_space, **kwargs)

        assert issubclass(feature_type, Feature)

        # At this stage, no recursion is needed.
        # We expect a specific supported feature type is to be matched now.
        if not isinstance(concept, structure.BrainStructure):
            raise ValueError(
                f"{concept.__class__.__name__} cannot be used for feature queries as it is not a BrainStructure type."
            )

        # Collect any preconfigured instances of the requested feature type
        # which match the query concept
        instances = [
            instance
            for f_type in cls._SUBCLASSES[feature_type]
            for instance in f_type._get_instances()
        ]

        preconfigured_instances = [
            f for f in siibra_tqdm(
                instances,
                desc=f"Matching {feature_type.__name__} to {concept}",
                total=len(instances),
                disable=(not instances)
            )
            if f.matches(concept, restrict_space)
        ]

        # Then run any registered live queries for the requested feature type
        # with the query concept.
        live_instances = feature_type._livequery(concept, **kwargs)

        results = list(dict.fromkeys(preconfigured_instances + live_instances))
        return CompoundFeature._compound(results, concept)

    @classmethod
    def _get_instance_by_id(cls, feature_id: str, **kwargs):
        try:
            return CompoundFeature._get_instance_by_id(feature_id, **kwargs)
        except ParseCompoundFeatureIdException:
            pass

        try:
            F, concept, fid = cls._deserialize_query_context(feature_id)
            return [
                f
                for f in F._livequery(concept, **kwargs)
                if f.id == fid or f.id == feature_id
            ][0]
        except ParseLiveQueryIdException:
            candidates = [
                inst
                for Cls in Feature._SUBCLASSES[Feature]
                for inst in Cls._get_instances()
                if inst.id == feature_id
            ]
            if len(candidates) == 0:
                raise NotFoundException(f"No feature instance wth {feature_id} found.")
            if len(candidates) == 1:
                return candidates[0]
            else:
                raise RuntimeError(
                    f"Multiple feature instance match {feature_id}",
                    [c.name for c in candidates]
                )
        except IndexError:
            raise NotFoundException

    @staticmethod
    def _wrap_livequery_feature(feature: 'Feature', fid: str):
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


class Compoundable(ABC):
    """
    Base class for structures which allow compounding.
    Determines the necessary grouping and compounding attributes.
    """
    _filter_attrs = []  # the attributes to filter this instance of feature
    _compound_attrs = []  # `compound_key` has to be created from `filter_attributes`

    def __init_subclass__(cls, **kwargs):
        assert len(cls._filter_attrs) > 0, "All compoundable classes have to have `_filter_attrs` defined."
        assert len(cls._compound_attrs) > 0, "All compoundable classes have to have `_compound_attrs` defined."
        assert all(attr in cls._filter_attrs for attr in cls._compound_attrs), "`_compound_attrs` must be a subset of `_filter_attrs`."
        cls._indexing_attrs = [
            attr
            for attr in cls._filter_attrs
            if attr not in cls._compound_attrs
        ]
        return super().__init_subclass__(**kwargs)

    def __init__(self):
        assert all(hasattr(self, attr) for attr in self._filter_attrs), "`_filter_attrs` can only consist of the attributes of the class."

    @property
    def filter_attributes(self) -> Dict[str, Any]:
        """
        Attributes that help distinguish or combine features of the same type
        among others.
        """
        return {attr: getattr(self, attr) for attr in self._filter_attrs}

    @property
    def _compound_key(self) -> Tuple[Any]:
        """
        A tuple of values that define the basis for compounding elements of
        the same type.
        """
        return tuple([self.filter_attributes[attr] for attr in self._compound_attrs])

    @property
    def _element_index(self) -> Any:
        """
        Unique index of this compoundable feature as an element of the Compound.
        Must be hashable.
        """
        index_ = [self.filter_attributes[attr] for attr in self._indexing_attrs]
        index = index_[0] if len(index_) == 1 else tuple(index_)
        assert hash(index), "`_element_index` of a compoundable must be hashable."
        return index

    @classmethod
    def _merge_anchors(cls, anchors: List[_anchor.AnatomicalAnchor]):
        return sum(anchors)


class CompoundFeature(Feature):
    """
    A compound aggregating mutliple features of the same type, forming its
    elements. The anatomical anchors and data of the features is merged.
    Features need to subclass "Compoundable" to allow aggregation
    into a compound feature.
    """

    def __init__(
        self,
        elements: List['Feature'],
        queryconcept: Union[region.Region, parcellation.Parcellation, space.Space]
    ):
        """
        A compound of several features of the same type with an anchor created
        as a sum of adjoinable anchors.
        """
        self._feature_type = elements[0].__class__
        assert all(isinstance(f, self._feature_type) for f in elements), NotImplementedError("Cannot compound features of different types.")
        self.category = elements[0].category  # same feature types have the same category
        assert issubclass(self._feature_type, Compoundable), NotImplementedError(f"Cannot compound {self._feature_type}.")

        modality = elements[0].modality
        assert all(f.modality == modality for f in elements), NotImplementedError("Cannot compound features of different modalities.")

        compound_keys = {element._compound_key for element in elements}
        assert len(compound_keys) == 1, ValueError(
            "Only features with identical compound_key can be aggregated."
        )
        self._compounding_attributes = {
            attr: elements[0].filter_attributes[attr]
            for attr in elements[0]._compound_attrs
        }

        self._elements = {f._element_index: f for f in elements}
        assert len(self._elements) == len(elements), RuntimeError(
            "Element indices should be unique to each element within a CompoundFeature."
        )

        Feature.__init__(
            self,
            modality=modality,
            description="\n".join({f.description for f in elements}),
            anchor=self._feature_type._merge_anchors([f.anchor for f in elements]),
            datasets=list(dict.fromkeys([ds for f in elements for ds in f.datasets]))
        )
        self._queryconcept = queryconcept

    def __getattr__(self, attr: str) -> Any:
        """Expose compounding attributes explicitly."""
        if attr in self._compounding_attributes:
            return self._compounding_attributes[attr]
        if hasattr(self._feature_type, attr):
            raise AttributeError(
                f"{self.__class__.__name__} does not have access to '{attr}' "
                "since it does not have the same value for all its elements."
            )
        raise AttributeError(
            f"{self.__class__.__name__} or {self._feature_type.__name__} have no attribute {attr}."
        )

    def __dir__(self):
        return super().__dir__() + list(self._compounding_attributes.keys())

    def plot(self, *args, **kwargs):
        raise NotImplementedError(
            "CompoundFeatures does not have a standardized plot. Try plotting the elements instead."
        )

    @property
    def indexing_attributes(self) -> Tuple[str]:
        "The attributes determining the index of this CompoundFeature's elements."
        return tuple(self.elements[0]._indexing_attrs)

    @property
    def elements(self):
        """Features that make up the compound feature."""
        return list(self._elements.values())

    @property
    def indices(self):
        """Unique indices to features making up the CompoundFeature."""
        return list(self._elements.keys())

    @property
    def feature_type(self) -> Type:
        """Feature type of the elements forming the CompoundFeature."""
        return self._feature_type

    @property
    def name(self) -> str:
        """Returns a short human-readable name of this feature."""
        groupby = ', '.join([
            f"{v} {k}" for k, v in self._compounding_attributes.items()
        ])
        return (
            f"{self.__class__.__name__} of {len(self)} "
            f"{self.feature_type.__name__} features grouped by ({groupby})"
            f" anchored at {self.anchor}"
        )

    @property
    def id(self) -> str:
        return "::".join((
            "cf0",
            f"{self._feature_type.__name__}",
            self._encode_concept(self._queryconcept),
            self.datasets[0].id if self.datasets else "nodsid",
            md5(self.name.encode("utf-8")).hexdigest()
        ))

    def __iter__(self) -> Iterator['Feature']:
        """Iterate over subfeatures"""
        return self.elements.__iter__()

    def __len__(self):
        """Number of subfeatures making the CompoundFeature"""
        return len(self._elements)

    def __getitem__(self, index: Any):
        """Get the nth element in the compound."""
        return self.elements[index]

    def get_element(self, index: Any):
        """Get the element with its unique index in the compound."""
        try:
            return self._elements[index]
        except Exception:
            raise IndexError(f"No feature with index '{index}' in this compound.")

    @classmethod
    def _compound(
        cls,
        features: List['Feature'],
        queryconcept: Union[region.Region, parcellation.Parcellation, space.Space]
    ) -> List['CompoundFeature']:
        """
        Compound features of the same the same type based on their `_compound_key`.

        If there are features that are not of type `Compoundable`, they are
        returned as is.

        Parameters
        ----------
        features: List[Feature]
            Feature instances to be compounded.
        queryconcept:
            AtlasConcept used for the query.

        Returns
        -------
        List[CompoundFeature | Feature]
        """
        non_compound_features = []
        grouped_features = defaultdict(list)
        for f in features:
            if isinstance(f, Compoundable):
                grouped_features[f._compound_key].append(f)
                continue
            non_compound_features.append(f)
        return non_compound_features + [
            cls(fts, queryconcept)
            for fts in grouped_features.values() if fts
        ]

    @classmethod
    def _get_instance_by_id(cls, feature_id: str, **kwargs):
        """
        Use the feature id to obtain the same feature instance.

        Parameters
        ----------
        feature_id : str

        Returns
        -------
        CompoundFeature

        Raises
        ------
        ParseCompoundFeatureIdException
            If no or multiple matches are found. Or id is not fitting to
            compound features.
        """
        if not feature_id.startswith("cf0::"):
            raise ParseCompoundFeatureIdException("CompoundFeature id must start with cf0::")
        cf_version, clsname, *queryconcept, dsid, fid = feature_id.split("::")
        assert cf_version == "cf0"
        candidates = [
            f
            for f in Feature._match(
                concept=cls._decode_concept(queryconcept),
                feature_type=clsname
            )
            if f.id == fid or f.id == feature_id
        ]
        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            else:
                raise ParseCompoundFeatureIdException(
                    f"The query with id '{feature_id}' have resulted multiple instances.")
        else:
            raise ParseCompoundFeatureIdException

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        for idx, element in siibra_tqdm(self._elements.items(), desc="Exporting elements", unit="element"):
            if '/' in str(idx):
                logger.warning(f"'/' will be replaced with ' ' of the file for element with index {idx}")
            filename = '/'.join([
                str(i).replace('/', ' ')
                for i in (idx if isinstance(idx, tuple) else [idx])
            ])
            fh.writestr(f"{self.feature_type.__name__}/{filename}.csv", element.data.to_csv())
