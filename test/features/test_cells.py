import unittest
import siibra


atlas = siibra.atlases["human"]
region = siibra.get_region("julich 2.9", "hoc1 left")


class TestCorticalCellDistribution(unittest.TestCase):
    def test_get_hoc1_cortical_cell_distribution(self):
        features = siibra.features.get(
            region, siibra.features.cellular.LayerwiseCellDensity
        )
        assert len(features) > 0

    def test_check_average_density(self):
        features = siibra.features.get(
            region, siibra.features.cellular.LayerwiseCellDensity
        )
        assert len(features) > 0
        feature = features[0]
        assert (feature.data["mean"].mean() > 90) and (
            feature.data["mean"].mean() < 100
        )
        assert feature.data.shape == (6, 2)

    def test_check_layer_density_with_valid_range(self):
        features = siibra.features.get(
            region, siibra.features.cellular.LayerwiseCellDensity
        )
        assert len(features) > 0
        feature = features[0]
        layer_density = feature.data[feature.data["layername"] == "I"]["density"]
        assert layer_density is not None
        layer_density = feature.data[feature.data["layername"] == "VI"]["density"]
        assert layer_density is not None


if __name__ == "__main__":
    unittest.main()
