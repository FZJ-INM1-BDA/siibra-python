import unittest
import siibra


atlas = siibra.atlases['human']
region = atlas.get_region("hoc1 left", parcellation="2.9")


class TestCorticalCellDistribution(unittest.TestCase):

    def test_get_hoc1_cortical_cell_distribution(self):
        features = siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity)
        assert len(features) > 1

    def test_check_average_density(self):
        features = siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity)
        assert len(features) > 1
        feature = features[0]
        average_density = feature.average_density()
        assert average_density is not None

    def test_check_layer_density_with_valid_range(self):
        features = siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity)
        assert len(features) > 1
        feature = features[0]
        layer_density = feature.layer_density(1)
        assert layer_density is not None
        layer_density = feature.layer_density(6)
        assert layer_density is not None

    def test_check_layer_density_with_invalid_range(self):
        features = siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity)
        assert len(features) > 1
        feature = features[0]
        with self.assertRaises(AssertionError):
            feature.layer_density(10)

    def test_to_model(self):
        features = siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity)
        feature = features[0]
        model = feature.to_model(detail=False)
        assert getattr(model.metadata, 'short_name') is not None and model.metadata.short_name != ""


if __name__ == "__main__":
    unittest.main()
