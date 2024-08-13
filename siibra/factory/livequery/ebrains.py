from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from typing import Any, List
import itertools
import json
from typing import List, Union, Iterator

try:
    from typing import TypedDict, Literal
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict, Literal

from siibra.attributes import AttributeCollection

from .base import LiveQuery
from ...concepts import Feature
from ...retrieval.file_fetcher.dataproxy_fetcher import DataproxyRepository
from ...attributes.descriptions import (
    RegionSpec,
    EbrainsRef,
    Name,
    ID,
    Doi,
    Url,
    Version,
)
from ...attributes.descriptions.modality import Modality, register_modalities
from ...cache import fn_call_cache
from ...commons.logger import logger

filepath = "ebrainsquery/v3/{schema}/{id}.json"


def _get_v3_ebrains_ref(schema: Literal["Dataset", "DatasetVersion"], id: str):
    try:
        fp = filepath.format(schema=schema, id=id)
        resp_json = json.loads(EbrainsQuery.repo.get(fp))
        return schema, id, resp_json.get("fullName")
    except:
        return schema, id, "Missing in cache"


def _get_all_ebrains_refs():
    from ..configuration import Configuration

    conf = Configuration()
    for folder in ["spaces", "parcellations", "maps", "features"]:
        ds_set = set()
        dsv_set = set()
        for f in conf.iter_jsons(folder):
            for attr in f.get("attributes", []):
                if attr.get("@type") != "siibra/attr/desc/ebrains/v0.1":
                    continue
                ids = attr.get("ids")
                ds = ids.get("openminds/Dataset")
                if ds:
                    ds_set.add(ds)
                dsv = ids.get("openminds/DatasetVersion")
                if dsv:
                    dsv_set.add(dsv)

        with ThreadPoolExecutor() as exec:
            ds_output = list(
                tqdm(
                    exec.map(_get_v3_ebrains_ref, itertools.repeat("Dataset"), ds_set),
                    total=len(ds_set),
                )
            )

            dsv_output = list(
                tqdm(
                    exec.map(
                        _get_v3_ebrains_ref, itertools.repeat("DatasetVersion"), dsv_set
                    ),
                    total=len(dsv_set),
                )
            )

        with open(f"wp1_{folder}_dataset.json", "w") as fp:
            json.dump(ds_output, indent="\t", fp=fp)
            fp.write("\n")

        with open(f"wp1_{folder}_datasetversion.json", "w") as fp:
            json.dump(dsv_output, indent="\t", fp=fp)
            fp.write("\n")


# generated from https://json2pyi.pages.dev/#TypedDictInline
IsVersionOf = TypedDict("IsVersionOf", {"identifier": List[str], "id": str})
Author = TypedDict(
    "Author",
    {
        "givenName": str,
        "shortName": None,
        "familyName": str,
        "fullName": None,
        "type": List[str],
        "id": str,
    },
)
DoiTD = TypedDict(
    "Doi", {"schema.org/identifier": List[str], "identifier": str, "id": str}
)
Context = TypedDict("Context", {"@vocab": str})
DatasetVersion = TypedDict(
    "DatasetVersion",
    {
        "@context": Context,
        "doi": List[DoiTD],
        "author": List[Author],
        "custodian": List[Any],
        "type": List[str],
        "isVersionOf": List[IsVersionOf],
        "versionInnovation": str,
        "versionIdentifier": str,
        "homepage": str,
        "fullName": str,
        "description": None,
        "identifier": List[str],
        "id": str,
    },
)


ebrains_modality = Modality(value="ebrains datasets")


@register_modalities()
def register_ebrains_modality():
    yield ebrains_modality


class EbrainsQuery(LiveQuery[Feature], generates=Feature):
    repo = DataproxyRepository("reference-atlas-data")

    _BLACKLIST = {
        "Whole-brain parcellation of the Julich-Brain Cytoarchitectonic Atlas",
        "whole-brain collections of cytoarchitectonic probabilistic maps",
        "DiFuMo atlas",
        "Automated Anatomical Labeling (AAL1) atlas",
    }

    _PREFIX = "eb-dsv-"

    def __init__(self, input: List[AttributeCollection]):
        super().__init__(input)
        self.sess = requests.Session()

    @staticmethod
    def iter_ids(input: Union[str, List[str], None]) -> List[str]:
        if not input:
            return []
        if isinstance(input, str):
            return [input]
        if isinstance(input, list):
            assert all(
                (isinstance(i, str) for i in input)
            ), f"Expecting all entries to be str, but is not: {input}"
            return input
        raise RuntimeError(
            f"Can only work with Non, str, or List[str], but got {type(input)}"
        )

    @classmethod
    def get_dsv(cls, dsv: str) -> DatasetVersion:
        fp = filepath.format(schema="DatasetVersion", id=dsv)
        return json.loads(cls.repo.get(fp))

    @classmethod
    def get_dsv_from_pe(cls, pe: str):
        fp = filepath.format(schema="ParcellationEntity_studyTarget", id=pe)
        resp_json = json.loads(cls.repo.get(fp))

        study_targets = resp_json.get("studyTarget")

        result = []
        for st in study_targets:
            st_id: str = st.get("id")
            assert st_id, f"id of study target was not defined! pe: {pe}"

            if "https://openminds.ebrains.eu/core/DatasetVersion" in st.get("type"):
                stripped_st_id = st_id.split("/")[-1]
                result.append(EbrainsQuery.get_dsv(stripped_st_id))
                continue

            logger.warning(
                f"DatasetVersion is not the type of study target: {st.get('type')}, skipping"
            )
        return result

    @classmethod
    def get_dsv_from_pev(cls, pev: str):
        fp = filepath.format(schema="ParcellationEntityVersion_studyTarget", id=pev)
        resp_json = json.loads(cls.repo.get(fp))

        study_targets = resp_json.get("studyTarget")

        result = []
        for st in study_targets:
            st_id: str = st.get("id")
            assert st_id, f"id of study target was not defined! pev: {pev}"

            if "https://openminds.ebrains.eu/core/DatasetVersion" in st.get("type"):
                stripped_st_id = st_id.split("/")[-1]
                result.append(EbrainsQuery.get_dsv(stripped_st_id))
                continue

            logger.warning(
                f"DatasetVersion is not the type of study target: {st.get('type')}, skipping"
            )
        return result

    def generate(self) -> Iterator[Feature]:
        mods = [mod for mods in self.find_attributes(Modality) for mod in mods]
        if ebrains_modality not in mods:
            return

        # in some circumstances, regions in siibra is separated into left and right hemisphere, but the parcellation entity (PE)
        # equivalent is not in openminds. As a result, once we decode the region, we also include the parent if and only if parent's name
        # is in child's name
        _rr = [
            (rspec, region)
            for rspecs in self.find_attributes(RegionSpec)
            for rspec in rspecs
            for decoded_region in rspec.decode()
            for region in (
                [decoded_region, decoded_region.parent]
                if (
                    decoded_region.parent
                    and decoded_region.parent.name in decoded_region.name
                )
                else [decoded_region]
            )
        ]

        for rspec, region in _rr:
            for ref in region._find(EbrainsRef):
                for pe in EbrainsQuery.iter_ids(
                    ref.ids.get("openminds/ParcellationEntity")
                ):
                    for dsv in EbrainsQuery.get_dsv_from_pe(pe):
                        attributes = []

                        homepage = dsv.get("homepage")
                        if homepage:
                            attributes.append(Url(value=homepage))

                        for doi in dsv.get("doi", []):
                            attributes.append(Doi(value=doi["identifier"]))

                        id = dsv.get("id")
                        if id:
                            id = EbrainsQuery._PREFIX + id
                            attributes.append(ID(value=id))

                        attributes.append(rspec)

                        yield Feature(attributes=attributes)

                for pev in EbrainsQuery.iter_ids(
                    ref.ids.get("openminds/ParcellationEntityVersion")
                ):
                    for dsv in EbrainsQuery.get_dsv_from_pev(pev):
                        attributes = []

                        homepage = dsv.get("homepage")
                        if homepage:
                            attributes.append(Url(value=homepage))

                        for doi in dsv.get("doi", []):
                            attributes.append(Doi(value=doi["identifier"]))

                        id = dsv.get("id")
                        if id:
                            id = EbrainsQuery._PREFIX + id
                            attributes.append(ID(value=id))

                        attributes.append(rspec)

                        yield Feature(attributes=attributes)
