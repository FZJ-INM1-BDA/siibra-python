# Copyright 2018-2024
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

from dataclasses import dataclass, field, replace, asdict
from typing import (
    Tuple,
    Type,
    TypeVar,
    Iterable,
    Callable,
    List,
    Union,
    BinaryIO,
    Dict,
)
import pandas as pd
from zipfile import ZipFile
from collections import defaultdict

from .dataproviders import DataProvider
from .attribute import Attribute
from .locations import Location
from .descriptions import (
    Name,
    ID as _ID,
    Modality,
    Url,
    Doi,
    TextDescription,
    Facet,
    EbrainsRef,
    AttributeMapping,
)
from ..operations.tabular import RemapColRowDict, RenameColumnsAndOrRows
from ..commons.iterable import assert_ooo
from ..commons.logger import siibra_tqdm, logger

T = TypeVar("T")


MATRIX_INDEX_ENTITY_KEY = "x-siibra/matrix-index-entity/index"


def attr_of_general_interest(attr: Attribute):
    return MATRIX_INDEX_ENTITY_KEY in attr.extra


@dataclass
class AttributeCollection:
    schema: str = "siibra/attribute_collection"
    attributes: Tuple[Attribute] = field(default_factory=list, repr=False)

    # TODO consider if this is the best spot for populating
    def __post_init__(self):
        from .dataproviders.tabular import TabularDataProvider

        column_row_mapping: Dict[str, RemapColRowDict] = defaultdict(
            lambda: {"column_mapping": {}, "row_mapping": {}}
        )
        for attr_mapping in self._find(AttributeMapping):
            for regionname, mappings in attr_mapping.region_mapping.items():
                for mapping in mappings:
                    if mapping["@type"] != "csv/row-index":
                        continue
                    target = mapping.get("target", None)

                    row_col_index = mapping.get("index")
                    column_row_mapping[target]["column_mapping"][
                        row_col_index
                    ] = regionname
                    column_row_mapping[target]["row_mapping"][
                        row_col_index
                    ] = regionname
        if len(column_row_mapping) > 0:
            for tabulardata in self._find(TabularDataProvider):
                remap_ops = [
                    op
                    for op in tabulardata.ops
                    if op["type"] == RenameColumnsAndOrRows.type
                ]
                if len(remap_ops) > 0:
                    logger.debug("RenameRowColumn already mapped. Skipped. ")
                    continue

                remap_dict = column_row_mapping.get(
                    tabulardata.name
                ) or column_row_mapping.get(None)

                if remap_dict is None:
                    logger.debug(
                        "Cannot find a suitable col/row remapper for the following tabular data:",
                        tabulardata,
                    )
                tabulardata.append_op(
                    RenameColumnsAndOrRows.generate_specs(remap_dict=remap_dict)
                )

    def _get(self, attr_type: Type[T]):
        return assert_ooo(self._find(attr_type))

    def _find(self, attr_type: Type[T]):
        return list(self._finditer(attr_type))

    def _finditer(self, attr_type: Type[T]) -> Iterable[T]:
        for attr in self.attributes:
            if isinstance(attr, attr_type):
                yield attr

    def get_dataprovider(self, expr: str = None, index: Union[int, None] = None):
        if index is not None:
            assert expr is None, "Only index or expr can be set at a time."
            return self._find(DataProvider)[index]
        assert index is None, "Only index or expr can be set at a time."
        return assert_ooo(self.find_dataproviders(expr))

    def find_dataproviders(self, expr: str = None) -> List[DataProvider]:
        return list(self.data_providers_table.query(expr)["dataprovider"])

    @property
    def data_providers_table(self) -> pd.DataFrame:
        dataproviders = self._find(DataProvider)
        return pd.DataFrame(
            [
                {
                    "type": type(d).__name__,
                    "dataprovider": d,
                    **{
                        key: value
                        for key, value in asdict(d).items()
                        if key not in d.IGNORE_KEYS
                    },
                }
                for d in dataproviders
            ]
        )

    @property
    def volume_providers(self):
        from .dataproviders.volume import VolumeProvider

        return [attr for attr in self.attributes if isinstance(attr, VolumeProvider)]

    def filter(self, filter_fn: Callable[[Attribute], bool]):
        """
        Return a new `AttributeCollection` that is a copy of this one where the
        only the `Attributes` evaluating to True from `filter_fn` are collected.
        """
        return replace(
            self, attributes=tuple(attr for attr in self.attributes if filter_fn(attr))
        )

    @property
    def _attribute_mapping(self):
        try:
            return self._get(AttributeMapping)
        except Exception:
            logger.debug("Cannot fetch `_attribute_mapping`:", exec_info=1)
            return None

    @property
    def name(self):
        try:
            return self._get(Name).value
        except AssertionError:
            pass

    @property
    def ID(self):
        try:
            return self._get(_ID).value
        except AssertionError:
            pass

    @property
    def modalities(self):
        return [m.value for m in self._find(Modality)]

    @property
    def publications(self):
        from ..operations.doi_fetcher import get_citation

        citations = [
            Url(value=doi.value, text=get_citation(doi)) for doi in self._find(Doi)
        ]

        return [*self._find(Url), *citations]

    @property
    def description(self):
        text_descs = self._find(TextDescription)
        if len(text_descs) > 0:
            return text_descs[0].value
        ebrains_refs = self._find(EbrainsRef)
        for ref in ebrains_refs:
            ebrain_ref_descs = ref.descriptions
            if len(ebrain_ref_descs) > 0:
                return ebrain_ref_descs[0]

    # TODO deprecate. use data_providers_table
    @property
    def facets(self):
        df = pd.DataFrame(
            [{"key": facet.key, "value": facet.value} for facet in self._find(Facet)]
        )
        return pd.concat([df, *[attr.facets for attr in self.attributes]])

    # TODO deprecate
    @staticmethod
    def get_query_str(facet_dict=None, **kwargs):
        if facet_dict is not None:
            assert isinstance(facet_dict, dict), "positional argument must be a dict"
            assert all(
                (
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in facet_dict.items()
                )
            ), "Only string key value can be provided!"
        else:
            facet_dict = dict()

        return " | ".join(
            [
                f"key=='{key}' & value=='{value}'"
                for key, value in {
                    **facet_dict,
                    **kwargs,
                }.items()
            ]
        )

    # TODO deprecate, use find_dataproviders instead
    def filter_attributes_by_facets(self, facet_dict=None, **kwargs):
        """
        Return a new AttributeCollection, where the attributes either:

        - do not contain facet information OR
        - the attribute facet matches *ALL* of the specs
        """
        query_str = AttributeCollection.get_query_str(facet_dict, **kwargs)

        return self.filter(
            lambda a: (
                len(a.facets) == 0
                or (len(a.facets) > 0 and len(a.facets.query(query_str)) > 0)
            )
        )

    # TODO deprecate, use find_dataproviders instead
    @staticmethod
    def filter_facets(
        attribute_collections: List["AttributeCollection"], facet_dict=None, **kwargs
    ):
        query_str = AttributeCollection.get_query_str(facet_dict, **kwargs)
        return [
            ac for ac in attribute_collections if len(ac.facets.query(query_str)) > 0
        ]

    # TODO deprecate use find_dataproviders instead
    @staticmethod
    def find_facets(attribute_collections: List["AttributeCollection"]):
        return pd.concat([ac.facets for ac in attribute_collections])

    def relates_to(self, attribute_collection: Union["AttributeCollection", Location]):
        """Yields attribute from self, attribute from target attribute_collection, and how they relate"""
        from ..assignment import collection_qualify, preprocess_concept

        yield from collection_qualify(
            preprocess_concept(self), preprocess_concept(attribute_collection)
        )

    @property
    def ebrains_ids(self) -> Iterable[Tuple[str, str]]:
        """
        Yields all ebrains references as Iterable of Tuple, e.g.

        (
            ("openminds/ParcellationEntity", "foo"),
            ("minds/core/parcellationregion/v1.0.0", "bar"),
        )
        """
        from .descriptions import EbrainsRef

        for ebrainsref in self._finditer(EbrainsRef):
            for key, value in ebrainsref.ids.items():
                if isinstance(value, list):
                    for v in value:
                        yield key, v
                if isinstance(value, str):
                    yield key, value

    def to_zip(self, filelike: Union[str, BinaryIO]) -> None:
        """
        Save the attribute collection to a zip file. The exported zip file should contain a README.md, which contains:

        - the metadata associated with the attribute collection
        - a list of other files saved in the same zip file
        """
        from .descriptions.base import Description
        from .locations.base import Location
        from .dataproviders.base import DataProvider
        from .._version import __version__

        with ZipFile(filelike, "w") as fp:
            readme_md = f"""{self.__class__.__name__} (exported by siibra-python {__version__})"""
            filenum_counter = 0

            progress = siibra_tqdm(
                desc="Serializing attributes", total=len(self.attributes), unit="attr"
            )

            def process_attr(attr: Attribute):
                nonlocal readme_md, filenum_counter

                try:
                    for textdesc, suffix, binaryio in attr._iter_zippable():
                        if textdesc:
                            readme_md += f"\n\n---\n\n{textdesc}"

                        if binaryio:
                            filename = f"file{filenum_counter}{suffix or ''}"
                            filenum_counter += 1
                            readme_md += f"\n\nexported file: {filename}\n"
                            for colname, fac in attr.facets.iterrows():
                                readme_md += f"facets: ({fac['key']}={fac['value']})"

                            # TODO not ideal, since loads everything in memory. Ideally we can stream it as IO
                            fp.writestr(filename, binaryio.read())
                except Exception as e:
                    print(e, type(attr))
                    readme_md += f"\n\n---\n\nError processing {str(attr)}: {str(e)}"

            # Process desc attributes first
            for desc in self._finditer(Description):
                process_attr(desc)
                progress.update(1)

            # Process locations next
            for loc in self._finditer(Location):
                process_attr(loc)
                progress.update(1)

            # Process data last
            for data in self._find(DataProvider):
                process_attr(data)
                progress.update(1)

            progress.close()

            fp.writestr("README.md", readme_md)
