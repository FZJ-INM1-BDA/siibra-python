import json
from typing import TypedDict, List, Dict

from ...retrieval_new.file_fetcher import GitHttpRepository, TarRepository
from ...commons import logger, KeyAccessor
from ...concepts import Attribute, AttributeCollection
from ...atlases import Parcellation, Region, Space
from ...descriptions import Name, RGBColor, Url, SpeciesSpec, ID
from ...assignment import register_collection_generator, collection_match

url = "https://gin.g-node.org/BrainGlobe/atlases.git"
fileurl = "https://gin.g-node.org/BrainGlobe/atlases/raw/master/{filename}.tar.gz"

repo: GitHttpRepository = None


DESC_INFO = """
------------------------------------
DESC INFO ABOUT USING BRAINGLOBE API
------------------------------------
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


def _get_repo():
    global repo
    if repo is None:
        print(DESC_INFO)
        repo = GitHttpRepository(url=url)
    return repo


vocab = KeyAccessor()
_populated = False
_registered_atlas = set()


@vocab.register_dir_callback
def _populate():
    global _populated
    if _populated:
        return
    _populated = True

    repo = _get_repo()
    files = [
        obj.filename.replace(".tar.gz", "")
        for obj in repo.ls()
        if obj.filename.endswith(".tar.gz")
    ]
    for filename in files:
        vocab.register(filename)


def ls():
    return list(vocab.__dir__())


def populate_regions(
    structures: List[Structure],
    parcellation: Parcellation,
    additional_attrs: List[Attribute] = None,
):

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


def use(atlas_name: str):
    if atlas_name in _registered_atlas:
        logger.info(f"{atlas_name} is already loaded.")
        return

    assert atlas_name in vocab, f"{atlas_name} not found"
    repo = TarRepository(fileurl.format(filename=atlas_name), gzip=True)

    metadata: Metadata = json.loads(repo.get(f"{atlas_name}/metadata.json"))
    structures: List[Structure] = json.loads(repo.get(f"{atlas_name}/structures.json"))

    speciesspec = SpeciesSpec(value=metadata["species"])
    parcellation = Parcellation(
        attributes=[
            ID(value=f"bg-{atlas_name}"),
            Name(value=metadata["name"] + " Parcellation"),
            Url(value=metadata["atlas_link"]),
            speciesspec,
        ],
        children=[],
    )
    populate_regions(structures, parcellation, [speciesspec])

    @register_collection_generator(Parcellation)
    def bg_parcellations(filter_param: AttributeCollection):
        if collection_match(filter_param, parcellation):
            yield parcellation

    space = Space(
        attributes=[
            ID(value=f"bg-{atlas_name}"),
            Name(value=metadata["name"] + " Space"),
            Url(value=metadata["atlas_link"]),
            speciesspec,
        ]
    )

    @register_collection_generator(Space)
    def bg_spaces(filter_param: AttributeCollection):
        if collection_match(filter_param, space):
            yield space

    _registered_atlas.add(atlas_name)
    logger.info(f"{atlas_name} added to siibra.")

    return space, parcellation
