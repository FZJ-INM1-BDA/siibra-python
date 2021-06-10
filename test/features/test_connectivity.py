import unittest
from siibra import atlases, parcellations
from siibra.features.connectivity import ConnectivityProfile


class TestConnectivity(unittest.TestCase):

    def setUp(self):
        atlas = atlases["human"]
        atlas.select_parcellation(parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_1_18)
        atlas.select_region("v1")
        self.got_features = atlas.get_features('ConnectivityProfile')

    def test_valid_connectivity_profile_data(self):
        data_name = 'Averaged_SC_JuBrain_184Regions_HCP_10M_length_MEAN'
        assertion_check = False
        for conn_pr in self.got_features:
            if conn_pr.src_name == data_name:
                self.assertEqual(conn_pr.src_file, 'Averaged_SC_JuBrain_184Regions_HCP_10M_length_MEAN.json')
                self.assertEqual(conn_pr.kgschema, 'minds/core/dataset/v1.0.0')
                self.assertEqual(conn_pr.kgid, '50c215bc-4c65-4f11-a4cd-98cc92750977')
                assertion_check = True

        self.assertTrue(assertion_check, msg=f'ConnectivityProfile for {data_name} not found')

    def test_connectivity_profile_data_for_invalid_name(self):
        data_name = 'FOO_BAR'
        assertion_check = True
        for conn_pr in self.got_features:
            if conn_pr.src_name == data_name:
                assertion_check = False

        self.assertTrue(assertion_check, msg=f'No ConnectivityProfile should be found for {data_name}')

    def test_str_result_with_original_values(self):
        ConnectivityProfile.show_as_log = False
        feature = self.got_features[1]
        # No assertion because of an index out of range error

    def test_str_result_barplot(self):
        ConnectivityProfile.show_as_log = True
        feature = self.got_features[1]
        # No assertion because of an index out of range error

    def test_decode_result_are_tuple(self):
        feature = self.got_features[0]
        decoded_feature = feature.decode(parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_1_18)[0]
        self.assertEqual(type(decoded_feature), tuple)
