import pytest
import siibra

# We get all registered subclasses of Feature
@pytest.mark.parametrize('Cls', [Cls for Cls in siibra.features.Feature.SUBCLASSES[siibra.features.Feature]])
def test_get_instances(Cls: siibra.features.Feature):
    instances = Cls.get_instances()
    assert isinstance(instances, list)