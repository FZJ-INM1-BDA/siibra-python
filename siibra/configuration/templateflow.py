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

    def parcellation_spec(self):
        if self.siibra_type == "map":
            return self.entities.get("atlas", self.space.removeprefix("tpl-"))
        raise ValueError("Parcellation spec is only applicable to `map` types.")

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

    def find_parcellations(self, space: str):
        assert space in self.spaces
        return sorted(list({
            f.parcellation_spec()
            for f in self.ls_map_files(space)
        }))

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
        maptype: Union[Literal["labelled", "statistical"], None] = None,
    ) -> List[TemplateFlowFile]:
        return list(
            f
            for f in self._files
            if f.siibra_type == "map"
            and (space is None or f.space == space)
            and (maptype is None or f.modality == maptype)
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
                "variant": tf_file.volume_spec(),
                "providers": {
                    tf_file.provider_type: tf_file.url,
                },
            }
            for tf_file in self.ls_template_files(space)
        ]
        config = {
            "@type": "siibra/space/v0.0.1",
            "@id": "minds/core/referencespace/v1.0.0/" f"templateflow/{uuid}",
            "name": f"[TemplateFlow] {name}",
            "shortName": f"TemplateFlow: {space}",
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

    def create_map_config(self, space: str, parcellation_spec: str):
        pass

    def get_description(self, space: str) -> dict:
        desc_file = f"{space}/template_description.json"
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
        except ValueError:
            continue
        with open(
            f"{tf_config_path}/spaces/{filename}",
            mode="wt",
            encoding="utf-8",
        ) as fp:
            json.dump(space_conf, indent="\t", fp=fp)
            fp.write("\n")

    return LocalFileRepository(tf_config_path)
