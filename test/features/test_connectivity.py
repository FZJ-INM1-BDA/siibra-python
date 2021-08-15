import unittest
import siibra
from siibra.features.connectivity import ConnectivityProfile


class TestSwitchParc(unittest.TestCase):
    def test_check_switch_parc(self):
        atlas = siibra.atlases["human"]
        parcellation = atlas.get_parcellation("2.9")
        region = atlas.get_region("v1", parcellation=parcellation)
        v29_v1_feats = siibra.get_features(region, "ConnectivityProfile")
        # TODO enable once v29 conn data is integrated
        # self.assertTrue(len(v29_v1_feats) > 0)
        self.assertTrue(
            all([f.regionspec.parcellation.id == parcellation.id for f in v29_v1_feats])
        )

        parcellation = atlas.get_parcellation("1.18")
        region = atlas.get_region("hoc1 left", parcellation=parcellation)
        v118_hoc1_left_features = siibra.get_features(region, "ConnectivityProfile")
        self.assertTrue(len(v118_hoc1_left_features) > 0)
        self.assertTrue(
            all(
                [
                    f.regionspec.parcellation.id == parcellation.id
                    for f in v118_hoc1_left_features
                ]
            )
        )


class TestConnectivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atlas = siibra.atlases["human"]
        parcellation = atlas.get_parcellation(
            siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_1_18
        )
        cls.parc = siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_1_18
        region = atlas.get_region("v1", parcellation=parcellation)
        cls.got_features = siibra.get_features(region, "ConnectivityProfile")

    def test_valid_connectivity_profile_data(self):
        data_name = "Averaged_SC_JuBrain_184Regions_HCP_10M_length_MEAN"
        assertion_check = False
        for conn_pr in self.got_features:
            if conn_pr.name == data_name:
                self.assertEqual(
                    conn_pr.name, "Averaged_SC_JuBrain_184Regions_HCP_10M_length_MEAN"
                )
                self.assertEqual(conn_pr._matrix.type_id, "minds/core/dataset/v1.0.0")
                self.assertEqual(
                    conn_pr._matrix.id, "50c215bc-4c65-4f11-a4cd-98cc92750977"
                )
                assertion_check = True

        self.assertTrue(
            assertion_check, msg=f"ConnectivityProfile for {data_name} not found"
        )

    def test_connectivity_profile_data_for_invalid_name(self):
        data_name = "FOO_BAR"
        assertion_check = True
        for conn_pr in self.got_features:
            if conn_pr.name == data_name:
                assertion_check = False

        self.assertTrue(
            assertion_check,
            msg=f"No ConnectivityProfile should be found for {data_name}",
        )

    def test_str_result_with_original_values(self):
        ConnectivityProfile.show_as_log = False
        specific_feats = [
            feat
            for feat in self.got_features
            if feat.name == "Averaged_FC_JuBrain_184Regions_HCP_REST_FIX_AVER_MEAN"
            and self.parc in feat._matrix.parcellations
            # use left hemisphere
            and "left" in str(feat.regionspec)
        ]

        # from https://jugit.fz-juelich.de/t.dickscheid/brainscapes-datafeatures/-/blob/c88ed1f39f22592f499045d842060fc9e1f9685f/Averaged_FC_JuBrain_184Regions_HCP_REST_FIX_AVER_MEAN.json
        json_field_names = [
            "Area 4p (PreCG) - left hemisphere",
            "Area PGa (IPL) - left hemisphere",
            "Area hIP2 (IPS) - left hemisphere",
            "Area s24 (sACC) - left hemisphere",
            "Area hOc6 (POS) - left hemisphere",
            "Area OP3 (POperc) - left hemisphere",
            "Area 5L (SPL) - left hemisphere",
            "CA (Hippocampus) - left hemisphere",
            "Area hOc4v (LingG) - left hemisphere",
            "Interposed Nucleus (Cerebellum) - left hemisphere",
            "Area 6d2 (PreCG) - left hemisphere",
            "Area FG2 (FusG) - left hemisphere",
            "Area 25 (sACC) - left hemisphere",
            "Area hIP5 (IPS) - left hemisphere",
            "Area TE 1.0 (HESCHL) - left hemisphere",
            "Area PFt (IPL) - left hemisphere",
            "Area hOc4d (Cuneus) - left hemisphere",
            "Area 33 (ACC) - left hemisphere",
            "Area p24ab (pACC) - left hemisphere",
            "Area 7M (SPL) - left hemisphere",
            "Area OP4 (POperc) - left hemisphere",
            "Area PFm (IPL) - left hemisphere",
            "Area 6d3 (SFS) - left hemisphere",
            "Area hOc2 (V2, 18) - left hemisphere",
            "Area hIP1 (IPS) - left hemisphere",
            "Ventral Dentate Nucleus (Cerebellum) - left hemisphere",
            "Ch 4 (Basal Forebrain) - left hemisphere",
            "Area 7P (SPL) - left hemisphere",
            "Area hOc4la (LOC) - left hemisphere",
            "Area hOc5 (LOC) - left hemisphere",
            "CM (Amygdala) - left hemisphere",
            "Area 1 (PostCG) - left hemisphere",
            "Area Id7 (Insula) - left hemisphere",
            "Area hOc3v (LingG) - left hemisphere",
            "Area Id1 (Insula) - left hemisphere",
            "Area OP9 (Frontal Operculum) - left hemisphere",
            "Area Ig1 (Insula) - left hemisphere",
            "Area 7A (SPL) - left hemisphere",
            "Area OP1 (POperc) - right hemisphere",
            "Area 5Ci (SPL) - left hemisphere",
            "Area PFop (IPL) - left hemisphere",
            "Area 2 (PostCS) - left hemisphere",
            "Dorsal Dentate Nucleus (Cerebellum) - left hemisphere",
            "VTM (Amygdala) - left hemisphere",
            "Area OP2 (POperc) - left hemisphere",
            "Area 3a (PostCG) - left hemisphere",
            "Area Ig2 (Insula) - left hemisphere",
            "Area 6mp (SMA, mesial SFG) - left hemisphere",
            "Area TE 1.1 (HESCHL) - left hemisphere",
            "MF (Amygdala) - left hemisphere",
            "Area PFcm (IPL) - left hemisphere",
            "Area 5M (SPL) - left hemisphere",
            "SF (Amygdala) - left hemisphere",
            "IF (Amygdala) - left hemisphere",
            "Area hIP7 (IPS) - right hemisphere",
            "Area Fo2 (OFC) - left hemisphere",
            "Area hOc1 (V1, 17, CalcS) - right hemisphere",
            "Area 44 (IFG) - left hemisphere",
            "Area p32 (pACC) - left hemisphere",
            "Area hOc3d (Cuneus) - left hemisphere",
            "Area p24c (pACC) - left hemisphere",
            "Area hIP3 (IPS) - left hemisphere",
            "Area 45 (IFG) - left hemisphere",
            "Area 4a (PreCG) - left hemisphere",
            "Area OP8 (Frontal Operculum) - left hemisphere",
            "Area 6ma (preSMA, mesial SFG) - left hemisphere",
            "LB (Amygdala) - left hemisphere",
            "Area hIP8 (IPS) - left hemisphere",
            "Area Fp1 (FPole) - left hemisphere",
            "Area FG4 (FusG) - left hemisphere",
            "Area Fp2 (FPole) - left hemisphere",
            "Area FG3 (FusG) - left hemisphere",
            "Area 3b (PostCG) - left hemisphere",
            "Area PF (IPL) - left hemisphere",
            "Area 6d1 (PreCG) - left hemisphere",
            "Area Fo1 (OFC) - left hemisphere",
            "Area TE 1.2 (HESCHL) - left hemisphere",
            "Area Fo3 (OFC) - left hemisphere",
            "Subiculum (Hippocampus) - left hemisphere",
            "DG (Hippocampus) - left hemisphere",
            "Area TE 3 (STG) - left hemisphere",
            "Area hOc4lp (LOC) - left hemisphere",
            "Area hIP4 (IPS) - left hemisphere",
            "Area 7PC (SPL) - left hemisphere",
            "Area FG1 (FusG) - left hemisphere",
            "Area s32 (sACC) - left hemisphere",
            "Entorhinal Cortex - left hemisphere",
            "Area PGp (IPL) - left hemisphere",
            "HATA (Hippocampus) - left hemisphere",
            "Area hIP6 (IPS) - left hemisphere",
            "Area hPO1 (POS) - left hemisphere",
            "Area 4p (PreCG) - right hemisphere",
            "Area PGa (IPL) - right hemisphere",
            "Area hIP2 (IPS) - right hemisphere",
            "Area s24 (sACC) - right hemisphere",
            "Area hOc6 (POS) - right hemisphere",
            "Area OP3 (POperc) - right hemisphere",
            "Area 5L (SPL) - right hemisphere",
            "CA (Hippocampus) - right hemisphere",
            "Area hOc4v (LingG) - right hemisphere",
            "Interposed Nucleus (Cerebellum) - right hemisphere",
            "Area 6d2 (PreCG) - right hemisphere",
            "Area FG2 (FusG) - right hemisphere",
            "Area 25 (sACC) - right hemisphere",
            "Area hIP5 (IPS) - right hemisphere",
            "Area TE 1.0 (HESCHL) - right hemisphere",
            "Area PFt (IPL) - right hemisphere",
            "Area hOc4d (Cuneus) - right hemisphere",
            "Area 33 (ACC) - right hemisphere",
            "Area p24ab (pACC) - right hemisphere",
            "Area 7M (SPL) - right hemisphere",
            "Area OP4 (POperc) - right hemisphere",
            "Area PFm (IPL) - right hemisphere",
            "Area 6d3 (SFS) - right hemisphere",
            "Area hOc2 (V2, 18) - right hemisphere",
            "Area hIP1 (IPS) - right hemisphere",
            "Ventral Dentate Nucleus (Cerebellum) - right hemispher",
            "Ch 4 (Basal Forebrain) - right hemisphere",
            "Area 7P (SPL) - right hemisphere",
            "Area hOc4la (LOC) - right hemisphere",
            "Area hOc5 (LOC) - right hemisphere",
            "Ch 123 (Basal Forebrain) - both hemispheres",
            "CM (Amygdala) - right hemisphere",
            "Area 1 (PostCG) - right hemisphere",
            "Area Id7 (Insula) - right hemisphere",
            "Area hOc3v (LingG) - right hemisphere",
            "Area Id1 (Insula) - right hemisphere",
            "Area OP9 (Frontal Operculum) - right hemisphere",
            "Area Ig1 (Insula) - left hemisphere",
            "Area 7A (SPL) - right hemisphere",
            "Area OP1 (POperc) - right hemisphere",
            "Area 5Ci (SPL) - right hemisphere",
            "Area PFop (IPL) - right hemisphere",
            "Area 2 (PostCS) - right hemisphere",
            "Dorsal Dentate Nucleus (Cerebellum) - right hemisphere",
            "VTM (Amygdala) - right hemisphere",
            "Area OP2 (POperc) - right hemisphere",
            "Area 3a (PostCG) - right hemisphere",
            "Area Ig2 (Insula) - right hemisphere",
            "Area 6mp (SMA, mesial SFG) - right hemisphere",
            "Area TE 1.1 (HESCHL) - right hemisphere",
            "MF (Amygdala) - right hemisphere",
            "Area PFcm (IPL) - right hemisphere",
            "Area 5M (SPL) - right hemisphere",
            "SF (Amygdala) - right hemisphere",
            "IF (Amygdala) - right hemisphere",
            "Area hIP7 (IPS) - left hemisphere",
            "Area Fo2 (OFC) - right hemisphere",
            "Area hOc1 (V1, 17, CalcS) - left hemisphere",
            "Area 44 (IFG) - right hemisphere",
            "Area p32 (pACC) - right hemisphere",
            "Area hOc3d (Cuneus) - right hemisphere",
            "Area p24c (pACC) - right hemisphere",
            "Area hIP3 (IPS) - right hemisphere",
            "Area 45 (IFG) - right hemisphere",
            "Area 4a (PreCG) - right hemisphere",
            "Area OP8 (Frontal Operculum) - right hemisphere",
            "Area 6ma (preSMA, mesial SFG) - right hemisphere",
            "LB (Amygdala) - right hemisphere",
            "Area hIP8 (IPS) - right hemisphere",
            "Area Fp1 (FPole) - right hemisphere",
            "Area FG4 (FusG) - right hemisphere",
            "Area Fp2 (FPole) - right hemisphere",
            "Area FG3 (FusG) - right hemisphere",
            "Area 3b (PostCG) - right hemisphere",
            "Area PF (IPL) - right hemisphere",
            "Area 6d1 (PreCG) - right hemisphere",
            "Area Fo1 (OFC) - right hemisphere",
            "Area TE 1.2 (HESCHL) - right hemisphere",
            "Area Fo3 (OFC) - right hemisphere",
            "Subiculum (Hippocampus) - right hemisphere",
            "DG (Hippocampus) - right hemisphere",
            "Area TE 3 (STG) - right hemisphere",
            "Area hOc4lp (LOC) - right hemisphere",
            "Area hIP4 (IPS) - right hemisphere",
            "Area 7PC (SPL) - right hemisphere",
            "Area FG1 (FusG) - right hemisphere",
            "Area s32 (sACC) - right hemisphere",
            "Entorhinal Cortex - right hemisphere",
            "Area PGp (IPL) - right hemisphere",
            "Fastigial Nucleus (Cerebellum) - both hemispheres",
            "HATA (Hippocampus) - right hemisphere",
            "Area hIP6 (IPS) - right hemisphere",
            "Area hPO1 (POS) - right hemisphere",
        ]
        v1_lh = [
            0.35542981,
            0.18540488,
            0.22537384,
            0.03777338,
            0.45402415,
            0.3199495,
            0.3585776,
            0.14115108,
            0.62945453,
            0.07936991,
            0.25774742,
            0.49840018,
            0.03673088,
            0.36233614,
            0.29843108,
            0.29931034,
            0.64630742,
            0.06479634,
            0.13747455,
            0.46077507,
            0.39240186,
            0.23139665,
            0.22788796,
            0.77251882,
            0.24510135,
            0.08531242,
            0.09446368,
            0.4556998,
            0.48817883,
            0.3043917,
            0.06167007,
            0.36443348,
            0.22492578,
            0.668008,
            0.11667432,
            0.16534811,
            0.22990106,
            0.45365928,
            0.42714003,
            0.27802032,
            0.31959377,
            0.34597296,
            0.07495086,
            0.0096215,
            0.31434145,
            0.25644363,
            0.24366011,
            0.24072065,
            0.29902301,
            -0.00349905,
            0.38719626,
            0.43122873,
            0.03418435,
            0.01354333,
            0.46798328,
            0.03784834,
            0.83512862,
            0.2693219,
            0.22378185,
            0.68883302,
            0.14168174,
            0.33804613,
            0.20424882,
            0.35098221,
            0.23114579,
            0.2840059,
            0.05338196,
            0.43021141,
            0.21874113,
            0.33073417,
            0.17119284,
            0.32758105,
            0.34567961,
            0.32234083,
            0.26178823,
            0.01565376,
            0.31105153,
            0.17323184,
            0.11694365,
            0.13529079,
            0.32409975,
            0.53302098,
            0.41604605,
            0.32029756,
            0.52262383,
            0.08060916,
            0.09523539,
            0.31032895,
            0.03977978,
            0.28182463,
            0.54144246,
            0.34126642,
            0.25322467,
            0.27099502,
            0.03083176,
            0.40457996,
            0.30386228,
            0.37960325,
            0.17760679,
            0.62720724,
            0.06552326,
            0.27152857,
            0.45739291,
            0.01891762,
            0.36879212,
            0.25094611,
            0.3017666,
            0.65096755,
            0.08321652,
            0.12688119,
            0.4472655,
            0.42340184,
            0.29141836,
            0.24222705,
            0.81414253,
            0.23327786,
            0.08817994,
            0.13249754,
            0.48290029,
            0.50091385,
            0.32083566,
            0.02419098,
            0.05475177,
            0.39026253,
            0.17880463,
            0.68850442,
            0.14053255,
            0.23388943,
            0.14972411,
            0.45485603,
            0.41809811,
            0.31538724,
            0.29029602,
            0.33591392,
            0.09141269,
            0.0084816,
            0.28498346,
            0.21323903,
            0.19537125,
            0.231541,
            0.25822148,
            0.00196131,
            0.41752315,
            0.37341825,
            0.02457705,
            0.00091803,
            0.53826914,
            0.02283758,
            1,
            0.28286998,
            0.21758792,
            0.70406032,
            0.18306385,
            0.37656354,
            0.29736339,
            0.34948955,
            0.20071224,
            0.28675468,
            0.04093008,
            0.41029094,
            0.20474029,
            0.33618764,
            0.15605288,
            0.31025285,
            0.38235587,
            0.38442066,
            0.24802464,
            0.0236913,
            0.2130981,
            0.15031671,
            0.13137016,
            0.16162156,
            0.3330596,
            0.53562784,
            0.49469982,
            0.3381295,
            0.49297401,
            0.06523327,
            0.07574864,
            0.3341894,
            0.0791081,
            0.03522271,
            0.27347344,
            0.57904297,
        ]

        specific_feat = specific_feats[0]

        # access last element
        last_profile = specific_feat.profile[-1]
        self.assertTrue((last_profile - v1_lh[-1]) < 1e-3)

        # index accessor to last profile should have corresponding column name
        index = len(specific_feat.profile) - 1
        self.assertTrue(index <= len(specific_feat.regionnames))

        # last accessor points to last element in json column names
        regionname = specific_feat.regionnames[index]
        decoded_region = self.parc.decode_region(json_field_names[-1])
        self.assertTrue(decoded_region.matches(regionname))

    def test_decode_result_are_tuple(self):
        decoded_features = [
            feature.decode(
                siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_1_18,
                force=True,
            )[0]
            for feature in self.got_features
        ]
        self.assertTrue(
            all(
                [type(decoded_feature) == tuple for decoded_feature in decoded_features]
            )
        )

    def test_regionnames_dict(self):
        self.assertTrue(
            all([type(feature.regionnames) == tuple for feature in self.got_features])
        )

    def test_feat_can_be_str(self):
        self.assertTrue(
            all([type(str(feature)) == str for feature in self.got_features])
        )

    def test_regionnames_str(self):
        feature = self.got_features[0]
        self.assertTrue(all([type(value) == str for value in feature.regionnames]))


if __name__ == "__main__":
    unittest.main()
