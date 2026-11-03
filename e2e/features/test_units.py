"""Every preconfigured tabular feature has a unit for each of its values."""
import pytest

from siibra.features.tabular.tabular import Tabular


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


FEATURES = [f for cls in (Tabular, *_all_subclasses(Tabular)) for f in cls._get_instances()]


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f"{type(f).__name__}-{f.id}")
def test_units_match_data(feature: Tabular):
    data = feature.data
    specs = feature._units.resolved()
    keys = [k for k in specs if k != "*"]

    stray = [k for k in keys if k not in data.columns and k not in data.index]
    assert not stray, f"unit entries name no column or row: {stray}"

    if "*" not in specs:
        by_row = any(k in data.index for k in keys)
        values = data.index if by_row else data.select_dtypes("number").columns
        missing = [v for v in values if feature.unit_of(v) is None]
        assert not missing, f"values without a unit: {missing}"

    for label, unit in data.attrs["units"].items():
        assert unit == feature.unit_of(label)
