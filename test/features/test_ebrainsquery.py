from typing import List
import unittest
import pytest

import siibra

siibra.livequeries.ebrains.EbrainsFeatureQuery.COMPACT_FEATURE_LIST = False


class TestEbrainsQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
        region = atlas.get_region("hoc1 left")
        cls.feat = siibra.features.get(
            region, siibra.features.dataset.EbrainsDataFeature
        )

    def test_some_result_returned(self):
        assert len(self.feat) > 0

    def test_no_duplicates_returned(self):
        ids = [f.id for f in self.feat]
        assert len(self.feat) == len(list(set(ids)))


parameter = [
    (
        "waxholm v3",
        "neocortex hippocampus",
        {
            "exclude": [
                # buggy. hippocampus + difumo 512 is a big issue
                "DiFuMo atlas (512 dimensions)"
            ],
            "include": [
                # some of these still clearly doesn't look right
                # for e.g. some v1/v2 are in here
                # but one step at a time...
                "Large scale multi-channel EEG in rats",
                "Wistar rat hippocampus CA1 pyramidal cell morphologies – Extension with additional reconstructions",
                "Auditory stimulation during the sleep-wake cycle in the freely moving rat",
                "3D high resolution SRXTM image data of cortical vasculature of rat brain.",
                "Density measurements of different receptors for CA1 (Hippocampus) [rat, v2.0]",
                "Visualization of projections from insular cortex in rat with the use of anterograde tracers",
                "Density measurements of different receptors for CA, stratum moleculare (Hippocampus) [rat, v2.0]",
                "Density measurements of different receptors for CA2 (Hippocampus) [rat, v2.0]",
                "PCI-like measure in rodents",
                "Density measurements of different receptors for CA3 (Hippocampus) [rat, v2.0]",
                "Dose-dependent effects of ketamine on spontaneous and evoked EEG activity in rats",
                "Detailed dynamical laminar organisation in different cortical areas (in rats in vivo)",
                "Density measurements of different receptors for CA, stratum cellulare (Hippocampus) [rat, v2.0]",
                "3D reconstructions of pyramidal cells in rat hippocampal CA1 region",
                "Electrophysiological data of cortical layer 6 neurons and synaptically coupled neuronal pairs",
                "Density measurements of different receptors for DG (Hippocampus) [rat, v1.0]",
                "Test of consciousness metrics in rodents",
                "Morphological data of cortical layer 6 neurons and synaptically coupled neuronal pairs",
                "Visualization of projections from posterior parietal cortex in rat with the use of anterograde tracers",
                "Immunofluorescence data of cortical layer 6 neurons",
                "Density measurements of different receptors for DG (Hippocampus) [rat, v2.0]",
                "Graphical representation of rat cortical vasculature reconstructed from high resolution 3D SRXTM data.",
                "Density measurements of different receptors for CA, stratum cellulare (Hippocampus) [rat, v1.0]",
                "Wistar rat hippocampus CA1 pyramidal cell morphologies",
                "Density measurements of different receptors for CA3 (Hippocampus) [rat, v1.0]",
                "Density measurements of different receptors for CA1 (Hippocampus) [rat, v1.0]",
                "Density measurements of different receptors for CA, stratum moleculare (Hippocampus) [rat, v1.0]",
                "Density measurements of different receptors for CA2 (Hippocampus) [rat, v1.0]",
                "Multi-area recordings from visual and somatosensory cortices, perirhinal cortex and hippocampal CA1",
            ],
        },
    )
]


@pytest.mark.parametrize("parc_id,region_id,inc_exc", parameter)
def test_species(parc_id, region_id, inc_exc):
    parc: siibra.core.parcellation.Parcellation = siibra.parcellations[parc_id]
    r: siibra.core.region.Region = parc.get_region(region_id)
    features: List[siibra.features._basetypes.Feature] = siibra.features.get(
        r, siibra.features.dataset.EbrainsDataFeature
    )
    try:
        feature_names = [f.name for f in features]
    except siibra.retrieval.requests.MultiSourceRequestException:
        pytest.xfail(
            "Some ebrains dataset links are not working atm (see https://github.com/FZJ-INM1-BDA/siibra-python/issues/651)."
        )
    excludes: List[str] = inc_exc.get("exclude")
    for exc in excludes:
        assert exc not in feature_names

    includes: List[str] = inc_exc.get("include")
    for inc in includes:
        assert f"Ebrains Dataset: {inc}" in feature_names


if __name__ == "__main__":
    unittest.main()
