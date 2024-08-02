import pytest
import re
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

import siibra
from siibra.assignment.qualification import Qualification

regions = [
    ("julich 3.0", "Area 4p (PreCG) right"),
    ("julich 3.0", "hoc1 left"),
]


@pytest.mark.parametrize("parc_spec, region_spec", regions)
def test_region_spatial_props(parc_spec, region_spec):
    region = siibra.get_region(parc_spec, region_spec)
    props = region.get_components("mni152")
    # for idx, cmp in enumerate(props.components, start=1):
    #     assert cmp.volume >= props.components[idx - 1].volume


@pytest.mark.parametrize(
    "parc_spec, region_spec, result_len, check_regions",
    [
        ("julich 3.0.3", "julich 3.0.3", 1, ["Julich-Brain Cytoarchitectonic Atlas (v3.0.3)"]),
        (
            "waxholm 4",
            "lateral olfactory tract",
            2,
            ["lateral olfactory tract", "Nucleus of the lateral olfactory tract"],
        ),
        (
            "julich 2.9",
            "/area 4/i",
            12,
            ["Area 44 (IFG)", "Area 4p (PreCG) - right hemisphere", "Area 4a (PreCG) - left hemisphere"],
        ),
        (
            "Superficial fiber Bundles HCP",
            re.compile("rh_SF-SF_.*"),
            24,
            ["rh_SF-SF_9", "rh_SF-SF_19", "rh_SF-SF_23"],
        ),
        (
            "julich 3.0.3",
            "hoc1 left",
            1,
            ["Area hOc1 (V1, 17, CalcS) - left hemisphere"],
        ),
    ]
)
def test_parcellation_find(parc_spec, region_spec, result_len, check_regions):
    parc = siibra.parcellations.get(parc_spec)
    results = parc.find(region_spec)
    assert isinstance(results, list)
    assert len(results) == result_len
    result_names = [r.name for r in results]
    for check_region in check_regions:
        assert check_region in result_names, f"{check_region} expected, but not found"


@pytest.mark.parametrize("parc, reg_spec, has_related, has_homology, has_related_ebrains_reg", [
    ("julich 2.9", "PGa", True, True, False),
    ("monkey", "PG", False, True, False),
    ("waxholm v3", "cornu ammonis 1", True, False, True),
])
def test_homologies_related_regions(parc, reg_spec, has_related, has_homology, has_related_ebrains_reg):

    reg = siibra.get_region(parc, reg_spec)
    related_assessments = list(reg.get_related_regions())
    homology_assessments = [(reg1, reg2, qual) for reg1, reg2, qual in related_assessments if qual == Qualification.HOMOLOGOUS]
    other_v_assessments = [(reg1, reg2, qual) for reg1, reg2, qual in related_assessments if qual == Qualification.OTHER_VERSION]

    assert has_related == (len(other_v_assessments) > 0)
    assert has_homology == (len(homology_assessments) > 0)

    if has_related_ebrains_reg:
        with ThreadPoolExecutor() as ex:
            features = ex.map(
                siibra.find_features,
                [reg2 for reg1, reg2, qual in other_v_assessments],
                repeat("ebrains")
            )
        assert len([f for f in features]) > 0
