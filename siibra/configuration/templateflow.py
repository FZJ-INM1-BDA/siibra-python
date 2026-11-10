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
from __future__ import annotations

import pathlib
import hashlib
from urllib.parse import quote
from zipfile import ZipFile
import json
from typing import Union, Literal, List, Dict
from dataclasses import dataclass

import pandas as pd

from ..commons import logger, Species
from ..retrieval.cache import CACHE
from ..retrieval.requests import HttpRequest
from ..retrieval.repositories import LocalFileRepository

SKELETON_URL = (
    "https://raw.githubusercontent.com/templateflow/python-client/"
    "{revision}/templateflow/conf/templateflow-skel.zip"
)
TEMPLATEFLOW_S3 = "https://templateflow.s3.amazonaws.com"
MAP_EXTENSIONS = {
    "dseg": "labelled",
    "dparc": "labelled",
    "probseg": "statistical",
}
TEMPLATE_EXTENSIONS = {
    "T1w",
    "T2w",
    "T1map",
    "T2map",
    "T2star",
    "PD",
    "PDw",
    "FLAIR",
    "boldref",
    "epi",
    "UNIT1",
    "SPECT",
    "PET",
    "MP2RAGE",
    "MD",
    "FA",
    "veryinflated",
    "inflated",
    "sphere",
    "pial",
    "thickness",
    "midthickness",
    "flat",
    "sulc",
    "white",
    "roi",
    "curv",
    "myelinmap",
    "area",
    "mask",
}
SUPPORTED_FILE_FORMATS = {
    "nii.gz": "nii",
    "nii": "nii",
    "label.gii": "gii-label",
    "surf.gii": "gii-mesh",
    "gii": "gii-label",
}
TEMPLATEFLOW_CITATION = (
    "TemplateFlow: a community archive of imaging templates and atlases for "
    "improved consistency in neuroimaging R Ciric, R Lorenz, WH Thompson, M "
    "Goncalves, E MacNicol, CJ Markiewicz, YO Halchenko, SS Ghosh, KJ "
    "Gorgolewski, RA Poldrack, O Esteban bioRxiv 2021.02.10.430678; "
    "doi:10.1101/2021.02.10.430678"
)
TSV_INDEX_HEADERS = {
    "index",
    "Index",
    "label",
}
TSV_NAME_HEADERS = {
    "name",
    "structure",
}


@dataclass
class TemplateFlowFile:
    space: str
    cohort: Union[str, None]
    filepath: str
    entities: Dict[str, str]
    provider_type: str
    siibra_type: Literal["map", "template"]
    modality: str

    @property
    def url(self):
        return f"{TEMPLATEFLOW_S3}/" f"{quote(self.filepath, safe='/')}"

    @property
    def volume_spec(self):
        if self.siibra_type == "map":
            specs = [f"{k}: {v}" for k, v in self.entities.items() if k != "atlas"]
            return " - ".join(specs)
        if self.siibra_type == "template":
            specs = [f"{k}: {v}" for k, v in self.entities.items()]
            return " - ".join([self.modality] + specs)

        raise ValueError(f"Invalid siibra_type: {self.siibra_type}")


@dataclass
class TemplateFlow:
    revision: str = "master"

    def __post_init__(self):
        self._files: List["TemplateFlowFile"] = self.classify_templateflow_files()
        logger.info(
            (
                "Please cite TemplateFlow if you use it in your research:\n"
                f"{TEMPLATEFLOW_CITATION}"
            )
        )

    @property
    def skeleton_archive(self) -> ZipFile:
        req = HttpRequest(SKELETON_URL.format(revision=self.revision))
        return req.get()

    def classify_templateflow_files(self):
        files: List["TemplateFlowFile"] = []
        with self.skeleton_archive as archive:
            for f in archive.filelist:
                if not any(f.filename.endswith(s) for s in SUPPORTED_FILE_FORMATS):
                    continue

                parts = f.filename.split("/")
                for part in parts[:-1]:
                    if "cohort" in part.lower():
                        cohort = part
                        break
                else:
                    cohort = None

                fname = parts[-1]
                stem, *suffixes = fname.split(".")
                entities = stem.split("_")
                space = entities[0].removeprefix("tpl-")
                if "desc-" in entities[-1]:
                    bids_modality_suffix = None
                else:
                    bids_modality_suffix = entities[-1]
                entities = (
                    entities[1:] if bids_modality_suffix is None else entities[1:-1]
                )

                if bids_modality_suffix in MAP_EXTENSIONS:
                    siibra_type = "map"
                    modality_suffix = MAP_EXTENSIONS[bids_modality_suffix]
                elif bids_modality_suffix in TEMPLATE_EXTENSIONS:
                    siibra_type = "template"
                    modality_suffix = bids_modality_suffix
                else:
                    if fname.endswith("label.gii"):
                        siibra_type = "map"
                        modality_suffix = "labelled"
                    else:
                        logger.warning(f"Unknown BIDS suffix: {fname}")

                provider_type = SUPPORTED_FILE_FORMATS.get(".".join(suffixes))
                if provider_type is None and "gii" in suffixes:
                    provider_type = "gii-label"  # TODO: allow digesting other gii files
                files.append(
                    TemplateFlowFile(
                        space=space,
                        filepath=f.filename,
                        provider_type=provider_type,
                        entities={
                            ent.split("-")[0]: ent.split("-")[1] for ent in entities
                        },
                        siibra_type=siibra_type,
                        modality=modality_suffix,
                        cohort=cohort,
                    )
                )

        return files

    @property
    def spaces(self):
        return sorted(list(set(f.space for f in self._files)))

    def find_parcellations(self, space: Union[str, None] = None):
        assert space is None or space in self.spaces
        return sorted(
            list({f.entities.get("atlas", f.space) for f in self.ls_map_files(space)})
        )

    def _find_lut_files(
        self,
        space: str,
        parecellation: str,
        bids_map_type: Literal["dseg", "probseg"],
    ):
        tsvs = [
            fn
            for fn in self.skeleton_archive.namelist()
            if fn.endswith(f"{bids_map_type}.tsv")
            and fn.startswith(f"tpl-{space}")
            and parecellation in fn
        ]
        return tsvs

    def ls_template_files(
        self, space: Union[str, None] = None
    ) -> List[TemplateFlowFile]:
        return list(
            f
            for f in self._files
            if f.siibra_type == "template" and (space is None or f.space == space)
        )

    def ls_map_files(
        self,
        space: Union[str, None] = None,
        parcellation: Union[str, None] = None,
        map_type: Union[Literal["labelled", "statistical"], None] = None,
    ) -> List[TemplateFlowFile]:
        return list(
            f
            for f in self._files
            if f.siibra_type == "map"
            and (space is None or f.space == space)
            and (parcellation is None or parcellation in f.filepath)
            and (map_type is None or f.modality == map_type)
        )

    def _check_urls(self):
        import requests

        for f in self._files:
            req = requests.get(f.url, stream=True)
            try:
                req.raise_for_status()
            except Exception as e:
                print(e)

    @property
    def citation(self):
        return TEMPLATEFLOW_CITATION

    def create_space_config(self, space: str):
        description = self.get_description(space=space)
        species = description.get("Species")
        name = description.get("Name")
        try:
            Species.decode(species)
        except ValueError as e:
            logger.info(
                f"Species '{species}' is not yet supported by siibra. Skipping '{name}'"
            )
            raise e

        uuid = hashlib.md5(f"{space}:{description}".encode("utf-8")).hexdigest()
        volumes = [
            {
                "@type": "siibra/volume/v0.0.1",
                "variant": tf_file.volume_spec,
                "providers": {
                    tf_file.provider_type: tf_file.url,
                },
            }
            for tf_file in self.ls_template_files(space)
        ]
        config = {
            "@type": "siibra/space/v0.0.1",
            "@id": f"minds/core/referencespace/v1.0.0/templateflow-{self.revision}/{uuid}",
            "name": f"[TemplateFlow] {name}",
            "shortName": f"[TemplateFlow] {space.removeprefix('tpl-')}",
            "modality": "MRI",
            "species": species,
            "volumes": volumes,
            "license": self.get_license(space=space),
            "description": (
                "This reference space was generated automatically by siibra "
                f"from the TemplateFlow template:\n'{description}'."
            ),
            "publications": self._publications_from_description(description),
        }
        return config

    def create_parc_and_map_configs(
        self,
        space: str,
        parcellation: str,
        map_type: Literal["labelled", "statistical"],
    ):
        tsv_mapping = self._find_maps(
            space=space,
            parcellation=parcellation,
            map_type=map_type,
        )

        def get_conf_base(tsv_entities: List[str]):
            sub_parc_name = "[TemplateFlow] " + " - ".join(tsv_entities)
            uuid = hashlib.md5(f"{space}:{sub_parc_name}".encode("utf-8")).hexdigest()
            return {
                "@type": "siibra/map/v0.0.1",
                "@id": f"siibra-map-v0.0.templateflow-{self.revision}_{uuid}",
                "name": sub_parc_name,
                "space": {"name": self.get_description(space=space)["Name"]},
                "parcellation": {"name": sub_parc_name},
                **(
                    {"represented_as:_sparsemap": True}
                    if map_type == "statistical"
                    else {}
                ),
            }

        configs = {}
        for tsv, meta in tsv_mapping.items():
            with self.skeleton_archive.open(tsv) as fp:
                lut = pd.read_csv(fp, sep="\t")

            for index_col in TSV_INDEX_HEADERS:
                if index_col in lut.columns:
                    break
            else:
                continue
            for region_col in TSV_NAME_HEADERS:
                if region_col in lut.columns:
                    break
            else:
                continue

            key = tuple(meta["entities"])
            configs[key] = {
                **get_conf_base(meta["entities"]),
                "volumes": [],
                "indices": {
                    getattr(row, region_col): []
                    for row in lut.itertuples()
                },
            }
            for v_idx, v in enumerate(meta["volumes"]):
                configs[key]["volumes"].append(
                    {
                        "@type": "siibra/volume/v0.0.1",
                        "providers": {"nii": v.url}
                    }
                )
                for row in lut.itertuples():
                    configs[key]["indices"][getattr(row, region_col)].append(
                        {"volume": v_idx, "label": getattr(row, index_col)}
                    )

        return configs

    def _find_maps(
        self,
        space: str,
        parcellation: str,
        map_type: Literal["labelled", "statistical"],
    ) -> Dict[str, List["TemplateFlowFile"]]:
        if map_type == "labelled":
            bids_map_type = "dseg"
        if map_type == "statistical":
            bids_map_type = "probseg"

        tsvs = [
            fn
            for fn in self.skeleton_archive.namelist()
            if fn.endswith(f"{bids_map_type}.tsv")
            and fn.startswith(f"tpl-{space}")
            and parcellation in fn
        ]
        if len(tsvs) == 0:
            raise ValueError

        map_files = self.ls_map_files(
            space=space, parcellation=parcellation, map_type=map_type
        )

        tsv_mapping = {}
        for tsv in tsvs:
            entities = tsv.split("/")[-1].split(".")[0].split("_")[1:]
            tsv_mapping[tsv] = {
                "entities": entities,
                "volumes": [
                    mf
                    for mf in map_files
                    if all(ent in mf.filepath for ent in entities)
                ],
            }

        return tsv_mapping

    def get_description(self, space: str) -> dict:
        desc_file = f"tpl-{space}/template_description.json"
        with self.skeleton_archive.open(desc_file, "r") as fp:
            description = json.load(fp=fp)
        return description

    def get_license(self, space: str) -> str:
        try:
            return self.skeleton_archive.read(f"{space}/LICENSE").decode()
        except KeyError:
            return self.get_description(space).get(
                "License", "No license information was found."
            )

    @staticmethod
    def _publications_from_description(description: dict) -> list[dict[str, str]]:
        references = description.get("ReferencesAndLinks", [])

        if isinstance(references, dict):
            references = references.values()
        elif isinstance(references, str):
            references = [references]

        return [
            {"url": reference}
            for reference in references
            if isinstance(reference, str)
            and reference.startswith(("https://", "http://"))
        ]


def create_local_repository(
    revision: str = "master",
    *,
    output_folder: Union[str, pathlib.Path, None] = None,
) -> LocalFileRepository:
    tf_config_path = pathlib.Path(
        f"{CACHE.folder}/TemplateFlow-{revision}"
        if output_folder is None
        else output_folder
    )
    tf_config_path.joinpath("spaces").mkdir(exist_ok=True, parents=True)
    tf_config_path.joinpath("maps").mkdir(exist_ok=True, parents=True)
    tf_config_path.joinpath("parcellations").mkdir(exist_ok=True, parents=True)
    tf = TemplateFlow(revision=revision)

    # spaces
    for space in tf.spaces:
        try:
            space_conf = tf.create_space_config(space)
            filename = f"TemplateFlow-{space}.json"
        except (ValueError, KeyError):
            continue
        with open(
            f"{tf_config_path}/spaces/{filename}",
            mode="wt",
            encoding="utf-8",
        ) as fp:
            json.dump(space_conf, indent="\t", fp=fp)
            fp.write("\n")

    # maps and parcellations
    for space in tf.spaces:
        continue
        parcs = tf.find_parcellations(space=space)
        for parc in parcs:
            for mt in ["labelled", "statistical"]:
                try:
                    map_confs = tf.create_parc_and_map_configs(
                        space=space, parcellation=parc, map_type=mt,
                    )
                except ValueError:
                    continue

            # parc_filename = f"TemplateFlow-{parc}.json"
            # with open(
            #     f"{tf_config_path}/maps/{parc_filename}",
            #     mode="wt",
            #     encoding="utf-8",
            # ) as fp:
            #     json.dump(parc_conf, indent="\t", fp=fp)
            #     fp.write("\n")

            for map_conf in map_confs:
                map_filename = f"{map_conf['name']}.json"
                with open(
                    f"{tf_config_path}/maps/{map_filename}",
                    mode="wt",
                    encoding="utf-8",
                ) as fp:
                    json.dump(map_conf, indent="\t", fp=fp)
                    fp.write("\n")

    return LocalFileRepository(tf_config_path)
