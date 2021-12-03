import pytest
from siibra import spaces
from siibra.core.region import Region
from siibra.commons import MapType

region_name = "Interposed Nucleus (Cerebellum) left"
kg_id = "658a7f71-1b94-4f4a-8f15-726043bbb52a"
parentname = "region_parent"

definition = {
    "name": region_name,
    "rgb": [170, 29, 10],
    "labelIndex": 251,
    "ngId": "jubrain mni152 v18 left",
    "children": [],
    "position": [-9205882, -57128342, -32224599],
    "datasets": [
        {
            "kgId": kg_id,
            "kgSchema": "minds/core/dataset/v1.0.0",
            "filename": "Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]",
        },
        {
            "@type": "fzj/tmp/volume_type/v0.0.1",
            "@id": "fzj/tmp/volume_type/v0.0.1/pmap",
            "space_id": spaces[0].id,
            "name": "Probabilistic map " + region_name,
            "map_type": "labelled",
            "volume_type": "nii",
            "url": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000001_jubrain-cytoatlas-Area-Ch-4_pub/4.2/Ch-4_l_N10_nlin2Stdcolin27_4.2_publicP_b92bf6270f6426059d719a6ff4d46aa7.nii.gz",
        },
    ],
}

parent_definition = {
    "name": parentname,
    "rgb": [170, 29, 10],
    "labelIndex": 251,
    "ngId": "jubrain mni152 v18 left",
    "children": [ definition ],
    "position": [-9205882, -57128342, -32224599],
    "originDatasets": [
        {
            "kgId": kg_id,
            "kgSchema": "minds/core/dataset/v1.0.0",
            "filename": "Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]",
        }
    ],
    "volumeSrc": {},
}

class Tmp:
    id='test'
    name='test-name'

    def __hash__(self):
        return hash(self.id)

output = Region.parse_legacy(parent_definition,parcellation_id='foo-bar')
parents = [ r for r in output if r.name == parentname ]
children = [ r for r in output if r.name == region_name ]

def test_init_from_legacy():
    assert len(parents) == 1
    assert len(children) == 1

def test_parent_has_no_parents():
    assert parents[0].parent is None
    assert parents[0].has_node_parent(parents[0]) is False


def test_parent_has_children():
    assert len(parents[0].children) > 0
    assert all([
        type(c) is Region for c in parents[0].children
    ])

def test_parent_includes_child():
    assert parents[0].includes(children[0])

# pydantic does not allow property setter
# see https://github.com/samuelcolvin/pydantic/issues/1577
# TODO discuss, use function instead?
@pytest.mark.xfail
def test_parent_include_after_reset_children():
    parents[0].children = []
    assert parents[0].includes(children[0]) is False

def test_region_include_self():
    assert parents[0].includes(parents[0])
    assert children[0].includes(children[0])

def test_find_child_region():
    regions = parents[0].find(region_name)
    assert regions is not None
    assert len(regions) == 1
    assert regions[0] is children[0]

def test_child_has_node_parent():
    assert children[0].has_node_parent(parents[0])

def test_child_find_child():
    assert len(children[0].find(parentname)) == 0

match_param = ("match_arg,expected", [
    ("Interposed Nucleus", True),
    ("Area 51", False),
    (children[0], True),
    (parents[0], False)
])

@pytest.mark.parametrize(*match_param)
def test_match_param(match_arg,expected):
    assert children[0].matches(match_arg) == expected

regional_map_param = ("region,space,map_type,expected", [
    (parents[0], spaces[0], MapType.LABELLED, None),
    (children[0], "wird space", MapType.LABELLED, None),
    (children[0], spaces[0], MapType.CONTINUOUS, None),
])

# TODO maybe NYI?
@pytest.mark.xfail
@pytest.mark.parametrize(*regional_map_param)
def test_regional_map(region: Region,space,map_type,expected):
    assert region.get_regional_map(space, map_type) == expected


if __name__ == "__main__":
    test_init_from_legacy()