from dataclasses import FrozenInstanceError
from unittest.mock import PropertyMock, patch

import pytest

import siibra
from siibra.configuration.factory import Factory
from siibra.core.assignment import Qualification
from siibra.features.anchor import AnatomicalAnchor
from siibra.features.feature import Compoundable, CompoundFeature
from siibra.features.tabular.point_distribution import InjectionSite, TracerDistribution

from .test_point_distribution import MOUSE_SPACE, make_spec, write_tables

TRACER_TYPE = "siibra/feature/point_distribution/tracer_distribution/v0.1"
INJECTION_POINT = {"@type": "siibra/location/point/v0.1", "coordinate": [1.0, 2.0, 3.0], "space": {"@id": MOUSE_SPACE}}


@pytest.fixture
def repo(tmp_path):
    return write_tables(tmp_path)


def tracer_spec(repo_folder, **overrides):
    return make_spec(repo_folder, **{"@type": TRACER_TYPE, **overrides})


def test_tracer_type_builds_tracer_distributions(repo):
    features = Factory.from_json(tracer_spec(repo))
    assert all(isinstance(f, TracerDistribution) for f in features)
    assert features[0].modality == "tracer distribution"
    assert len(features[0].data) == 2


def test_tracer_distributions_are_not_compounded(repo):
    features = Factory.from_json(tracer_spec(repo))
    assert not issubclass(TracerDistribution, Compoundable)
    assert CompoundFeature._compound(features, None) == features


def test_tracer_distribution_is_a_connectivity_feature():
    assert siibra.features.connectivity.TracerDistribution is TracerDistribution
    assert TracerDistribution.category == "connectivity"


def test_injection_site_has_the_exact_regions(repo):
    feature = Factory.from_json(tracer_spec(repo))[0]
    regions = {"m2 (v4)": Qualification.EXACT, "m2 (v3)": Qualification.EXACT, "MOs": Qualification.HOMOLOGOUS}
    with patch.object(AnatomicalAnchor, "regions", new_callable=PropertyMock, return_value=regions):
        site = feature.injection_site
    assert site == InjectionSite(regions=("m2 (v4)", "m2 (v3)"), location=None)


def test_injection_site_may_be_only_a_location(repo):
    spec = tracer_spec(repo, location=INJECTION_POINT)
    del spec["region"]
    site = Factory.from_json(spec)[0].injection_site
    assert site.regions == ()
    assert site.location.coordinate == (1.0, 2.0, 3.0)


def test_injection_site_is_immutable():
    with pytest.raises(FrozenInstanceError):
        InjectionSite().regions = ()


def test_an_injection_site_is_required(repo):
    spec = tracer_spec(repo)
    del spec["region"]
    with pytest.raises(ValueError, match="injection site"):
        Factory.from_json(spec)
