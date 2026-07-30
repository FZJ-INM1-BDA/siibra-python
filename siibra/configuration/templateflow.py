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

from ..commons import logger, Species
from ..retrieval.cache import CACHE
from ..retrieval.requests import HttpRequest
from ..retrieval.repositories import LocalFileRepository

SKELETON_URL = (
    "https://raw.githubusercontent.com/templateflow/python-client/"
    "{revision}/templateflow/conf/templateflow-skel.zip"
)
TEMPLATEFLOW_S3 = "https://templateflow.s3.amazonaws.com"
TEMPLATE_VARIANTS = {
    "T1w": "T1 weighted",
    "T2w": "T2 weighted",
    "T1map": "T1 map",
    "T2map": "T2 map",
    "T2star": "T2*",
    "PD": "proton density",
    "FLAIR": "FLAIR",
}

NIFTI_EXTENSIONS = ("nii.gz", "nii")

MAP_EXTENSIONS = {
    "dseg",
    "probseg",
}

TEMPLATE_EXTENSIONS = {
    "T1w",
    "T2w",
    "T1map",
    "T2map",
    "T2star",
    "PD",
    "FLAIR",
}

TEMPLATEFLOW_CITATION = (
    "TemplateFlow: a community archive of imaging templates and atlases for "
    "improved consistency in neuroimaging R Ciric, R Lorenz, WH Thompson, M "
    "Goncalves, E MacNicol, CJ Markiewicz, YO Halchenko, SS Ghosh, KJ "
    "Gorgolewski, RA Poldrack, O Esteban bioRxiv 2021.02.10.430678; "
    "doi:10.1101/2021.02.10.430678"
)


class TemplateFlow:

    @staticmethod
    def classify_templateflow_files(
        revision: str = "master",
    ) -> dict[str, dict[str, list[str]]]:
        """Classify TemplateFlow templates, maps, and map TSV files."""
        spaces = {}
        req = HttpRequest(SKELETON_URL.format(revision=revision))
        with req.get() as archive:
            assert isinstance(archive, ZipFile)

            cohort_divided = {
                f.filename.split("/")[0]: None
                for f in archive.filelist
                if f.is_dir() and "cohort" in f.filename
            }

            spaces = {}
            for f in archive.filelist:
                if not f.is_dir() or "scripts" in f.filename:
                    continue

                parts = f.filename.split("/")
                key = parts[0]
                if key in cohort_divided and "cohort" in parts[1]:
                    spaces[f"{key}/{parts[1]}"] = spaces[key]
                else:
                    desc_file = f.filename + "template_description.json"
                    with archive.open(desc_file, "r") as fp:
                        description = json.load(fp=fp)
                    try:
                        license = archive.read(f.filename + "LICENSE").decode()
                    except KeyError:
                        license = description.get("License", "No license information was found.")

                    spaces.setdefault(
                        key,
                        {
                            "license": license,
                            "description": description,
                            "templates": {},
                            "maps": [],
                            "tsvs": [],
                        },
                    )

            # remove top level key for templates divided by cohorts
            for key in cohort_divided.keys():
                spaces.pop(key)

            for f in archive.filelist:
                if f.is_dir():
                    continue

                parts = f.filename.split("/")
                key = parts[0]
                if key in cohort_divided and "cohort" in parts[1]:
                    key = f"{key}/{parts[1]}"

                parts = f.filename.split(".")
                suffix = ".".join(parts[1:])
                entities = parts[0].split("_")
                bids_extension = entities[-1]

                if suffix == "tsv":
                    if key in spaces:
                        spaces[key]["tsvs"].append(f.filename)
                    else:
                        for k in spaces.keys():
                            if k.startswith(key):
                                spaces[k]["tsvs"].append(f.filename)
                    continue

                if suffix not in NIFTI_EXTENSIONS:
                    continue

                if bids_extension in MAP_EXTENSIONS:
                    spaces[key]["maps"].append(f.filename)
                if bids_extension in TEMPLATE_EXTENSIONS:
                    variant = TEMPLATE_VARIANTS.get(bids_extension)
                    qualifiers = [
                        ent
                        for ent in entities[:-1]
                        if ent.startswith(("res-", "desc-"))
                    ]
                    if qualifiers:
                        variant += f" ({'_'.join(qualifiers)})"
                    spaces[key]["templates"][variant] = f.filename

        return spaces

    def __init__(self, revision: str = "master"):
        self.revision = revision
        self._skeleton_cache = self.classify_templateflow_files(revision=revision)
        logger.info((
            "Please cite TemplateFlow if you use it in your research:\n"
            f"{TEMPLATEFLOW_CITATION}"
        ))

    @property
    def citation(self):
        return TEMPLATEFLOW_CITATION

    @staticmethod
    def _get_url(path: str) -> str:
        return f"{TEMPLATEFLOW_S3}/" f"{quote(path, safe='/')}"

    def create_space_config(self, tf_key: str):
        description = self._skeleton_cache[tf_key]["description"]
        species = description.get("Species")
        name = description.get("Name")
        try:
            Species.decode(species)
        except ValueError as e:
            logger.info(
                f"Species '{species}' is not yet supported by siibra. Skipping '{name}'"
            )
            raise e

        uuid = hashlib.md5(str(description).encode("utf-8")).hexdigest()
        volumes = [
            {
                "@type": "siibra/volume/v0.0.1",
                "variant": variant,
                "providers": {
                    "nii": self._get_url(path),
                },
            }
            for variant, path in self._skeleton_cache[tf_key].get("templates").items()
        ]
        config = {
            "@type": "siibra/space/v0.0.1",
            "@id": "minds/core/referencespace/v1.0.0/" f"templateflow/{uuid}",
            "name": f"[TemplateFlow] {name}",
            "shortName": f"TemplateFlow: {tf_key}",
            "modality": "MRI",
            "species": species,
            "volumes": volumes,
            "license": self._skeleton_cache[tf_key].get("license"),
            "description": (
                "This reference space was generated automatically by siibra "
                f"from the TemplateFlow template:\n'{description}'."
            ),
            "publications": self._publications_from_description(description),
        }
        return config

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


def create_local_repository(revision: str = "master") -> LocalFileRepository:
    tf_config_path = pathlib.Path(f"{CACHE.folder}/TemplateFlow-{revision}")
    tf_config_path.joinpath("spaces").mkdir(exist_ok=True, parents=True)
    tf_config_path.joinpath("maps").mkdir(exist_ok=True, parents=True)
    tf_config_path.joinpath("parcellations").mkdir(exist_ok=True, parents=True)
    tf = TemplateFlow(revision=revision)
    for key in tf._skeleton_cache:
        try:
            space_conf = tf.create_space_config(key)
        except ValueError:
            continue
        with open(
            f"{tf_config_path}/spaces/{key.replace('/', '_')}.json",
            mode="wt",
            encoding="utf-8",
        ) as fp:
            json.dump(space_conf, indent="\t", fp=fp)
            fp.write("\n")

    return LocalFileRepository(tf_config_path)
