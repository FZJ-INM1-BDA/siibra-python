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


from typing import Iterator, List, Union, Dict
import re
import json
from pathlib import Path
from io import BytesIO
from hashlib import md5
import numpy as np
from PIL import Image as PILImage
import nibabel as nib

try:
    from typing import TypedDict, Literal
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict, Literal

from .base import LiveQuery
from ...commons.logger import logger
from ...commons.string import to_hex
from ...attributes import Attribute
from ...attributes.descriptions import Name, ID, Url, SpeciesSpec, ParcSpec, SpaceSpec
from ...attributes.dataproviders import ImageRecipe
from ...atlases import Space, ParcellationScheme, Map
from ...operations.file_fetcher import GitHttpRepository, TarRepository


def tiff_to_nii(tiff_bytes: bytes, affine: np.ndarray, cast_to_int=False) -> str:
    from ...cache import CACHE

    filename = CACHE.build_filename(md5(tiff_bytes).hexdigest(), ".nii")
    if Path(filename).exists():
        return filename

    tiff_img = PILImage.open(BytesIO(tiff_bytes))

    stack = []
    for i in range(tiff_img.n_frames):
        tiff_img.seek(i)
        stack.append(np.array(tiff_img))
    stacked_array = np.stack(stack, axis=-1)
    if cast_to_int:
        stacked_array = stacked_array.astype("int32")
    nii = nib.Nifti1Image(stacked_array, affine)
    nib.save(nii, filename)
    return filename


url = "https://gin.g-node.org/BrainGlobe/atlases.git"
fileurl = "https://gin.g-node.org/BrainGlobe/atlases/raw/master/{filename}.tar.gz"


repo: Union[GitHttpRepository, None] = None


DESC_INFO = """
-------------------------------
INFO ABOUT USING BRAINGLOBE API
-------------------------------

Please visit https://brainglobe.info/documentation/brainglobe-atlasapi/index.html for more information on BrainGlobeAPI.

Citation:

Claudi, F., Petrucco, L., Tyson, A. L., Branco, T., Margrie, T. W. and Portugues, R. (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases. Journal of Open Source Software, 5(54), 2668, https://doi.org/10.21105/joss.02668

"""


class Structure(TypedDict):
    structure: str
    id: int
    name: str
    structure_id_path: List[int]
    rgb_triplet: List[int]


class Metadata(TypedDict):
    name: str
    citation: str
    atlas_link: str
    species: str
    symmetric: bool
    resolution: List[float]
    orientation: str
    version: str
    shape: List[int]
    transform_to_bg: List[List[float]]
    additional_references: List


PREFIX = "bg:"


def get_id(name, type: Literal["space", "parcellation", "map"]):
    return f"{PREFIX}{name}:{type}"


def get_name(input: str):
    input = input.replace(PREFIX, "")
    return re.sub(r"(:space|:parcellatoin|:map)$", "", input)


def _get_repo():
    global repo
    if repo is None:
        print(DESC_INFO)
        repo = GitHttpRepository(url=url, branch="master")
    return repo


def list_all() -> List[str]:
    repo = _get_repo()
    return [
        PREFIX + obj.filename.replace(".tar.gz", "")
        for obj in repo.ls()
        if obj.filename.endswith(".tar.gz")
    ]


class SpaceLiveQuery(LiveQuery[Space], generates=Space):
    def generate(self) -> Iterator[Space]:
        ids = [
            id
            for ids in self.find_attributes(ID)
            for id in ids
            if id.value.startswith(PREFIX)
        ]
        if len(ids) == 0:
            logger.debug(f"no ID attribute start with {PREFIX}, skip.")
            return
        if len(ids) > 1:
            logger.warning(
                f"Expected one and only one ID attribute starting with {PREFIX}, but got {len(ids)}. Skipping"
            )
            return
        id_value = ids[0].value
        atlas_name = get_name(id_value)

        repo = TarRepository(fileurl.format(filename=atlas_name), gzip=True)

        metadata: Metadata = json.loads(repo.get(f"{atlas_name}/metadata.json"))
        affine = np.array(
            metadata.get("transform_to_bg") or metadata.get("trasform_to_bg")
        )
        ref_img_filename = tiff_to_nii(repo.get(f"{atlas_name}/reference.tiff"), affine)
        speciesspec = SpeciesSpec(value=metadata["species"])

        space_id = get_id(atlas_name, "space")
        yield Space(
            attributes=[
                ID(value=space_id),
                Name(value=metadata["name"] + " bg space"),
                Url(value=metadata["atlas_link"]),
                speciesspec,
                ImageRecipe(format="nii", url=ref_img_filename, space_id=space_id),
            ]
        )


class ParcellationLiveQuery(
    LiveQuery[ParcellationScheme], generates=ParcellationScheme
):
    @staticmethod
    def populate_regions(
        structures: List[Structure],
        parcellation: "ParcellationScheme",
        additional_attrs: List["Attribute"] = None,
    ):
        from ...attributes.descriptions import Name, RGBColor, ID
        from ...atlases import Region

        _dict_id_to_region: Dict[int, Region] = {}
        _dict_region_to_parent: Dict[Region, int] = {}
        regions: List[Region] = []

        # TODO we needed to go through twice because we cannot guarantee that
        # parents appear before children in the list.

        for structure in structures:
            name = Name(value=structure["name"])
            rgb_str = "".join(
                [hex(value)[:2].rjust(2, "0") for value in structure["rgb_triplet"]]
            )
            rgb = RGBColor(value=f"#{rgb_str}")
            _id = ID(value=f"bg-{structure['name']}")

            region = Region(
                attributes=[_id, name, rgb, *(additional_attrs or [])], children=[]
            )
            _dict_id_to_region[structure["id"]] = region

            try:
                _dict_region_to_parent[region] = structure["structure_id_path"][-2]
            except IndexError:
                region.parent = parcellation

            regions.append(region)

        for region in regions:
            if region not in _dict_region_to_parent:
                continue
            parent_id = _dict_region_to_parent[region]
            assert parent_id in _dict_id_to_region, f"{parent_id} not found!"
            parent_region = _dict_id_to_region[parent_id]
            region.parent = parent_region

    def generate(self) -> Iterator[ParcellationScheme]:
        ids = [
            id
            for ids in self.find_attributes(ID)
            for id in ids
            if id.value.startswith(PREFIX)
        ]
        if len(ids) == 0:
            logger.debug(f"no ID attribute start with {PREFIX}, skip.")
            return
        if len(ids) > 1:
            logger.warning(
                f"Expected one and only one ID attribute starting with {PREFIX}, but got {len(ids)}. Skipping"
            )
            return
        id_value = ids[0].value
        atlas_name = get_name(id_value)

        repo = TarRepository(fileurl.format(filename=atlas_name), gzip=True)

        metadata: Metadata = json.loads(repo.get(f"{atlas_name}/metadata.json"))
        structures: List[Structure] = json.loads(
            repo.get(f"{atlas_name}/structures.json")
        )
        speciesspec = SpeciesSpec(value=metadata["species"])

        parcellation_id = get_id(atlas_name, "parcellation")
        parcellation = ParcellationScheme(
            attributes=[
                ID(value=parcellation_id),
                Name(value=metadata["name"] + " bg parcellation"),
                Url(value=metadata["atlas_link"]),
                speciesspec,
            ],
            children=[],
        )
        self.populate_regions(structures, parcellation, [speciesspec])
        yield parcellation


class MapLiveQuery(LiveQuery[Map], generates=Map):
    def generate(self) -> Iterator[Map]:
        atlas_names = None

        parc_spec = [
            get_name(id.value)
            for ids in self.find_attributes(ParcSpec)
            for id in ids
            if id.value.startswith(PREFIX)
        ]
        space_spec = [
            get_name(id.value)
            for ids in self.find_attributes(ParcSpec)
            for id in ids
            if id.value.startswith(PREFIX)
        ]
        ids = [
            get_name(id.value)
            for ids in self.find_attributes(ID)
            for id in ids
            if id.value.startswith(PREFIX)
        ]

        atlas_names = {value for value in [*parc_spec, *space_spec, *ids]}

        if len(atlas_names) == 0:
            logger.debug(f"no ID attribute start with {PREFIX}, skip.")
            return
        if len(atlas_names) > 1:
            logger.warning(
                f"Expected one and only one ID attribute starting with {PREFIX}, but got {len(ids)}. Skipping"
            )
            return
        atlas_name = list(atlas_names)[0]
        repo = TarRepository(fileurl.format(filename=atlas_name), gzip=True)

        metadata: Metadata = json.loads(repo.get(f"{atlas_name}/metadata.json"))
        structures: List[Structure] = json.loads(
            repo.get(f"{atlas_name}/structures.json")
        )
        speciesspec = SpeciesSpec(value=metadata["species"])

        parcellation_id = get_id(atlas_name, "parcellation")
        space_id = get_id(atlas_name, "space")
        map_id = get_id(atlas_name, "map")

        # "trasform_to_bg" typo seen in allen human 500um
        affine = np.array(
            metadata.get("transform_to_bg") or metadata.get("trasform_to_bg")
        )
        annot_img_filename = tiff_to_nii(
            repo.get(f"{atlas_name}/annotation.tiff"), affine, True
        )

        _region_attributes = {
            structure["name"]: [
                {
                    "@type": "volume/ref",
                    "label": structure["id"],
                    "color": to_hex(structure["rgb_triplet"]),
                }
            ]
            for structure in structures
        }
        labelled_map_image = ImageRecipe(
            format="nii",
            url=annot_img_filename,
            space_id=space_id,
        )

        yield Map(
            parcellation_id=parcellation_id,
            space_id=space_id,
            maptype="labelled",
            attributes=(
                ID(value=map_id),
                Name(value=f"bg labelled-map {space_id} {parcellation_id}"),
                labelled_map_image,
                speciesspec,
            ),
            region_mapping=_region_attributes,
        )
