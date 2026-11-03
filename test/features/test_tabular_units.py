import json
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from siibra.configuration.factory import Factory
from siibra.features.tabular.cortical_profile import CorticalProfile
from siibra.features.tabular.tabular import Tabular, _Units

DATA = pd.DataFrame({"mean": [1.0, 2.0], "std": [0.1, 0.2], "label": ["a", "b"]}, index=["x", "y"])


def make_tabular(unit=None, data=DATA):
    return Tabular(description="d", modality="m", anchor=None, data=data, unit=unit)


def test_unit_of_default_and_override():
    t = make_tabular({"*": {"unit": "mm", "description": "length"}, "std": {"unit": "cm"}})
    assert t.unit_of("mean") == "mm"
    assert t.unit_of("std") == "cm"
    assert t.unit_of("x") == "mm"  # row labels work the same way
    assert t._units.spec_of("std") == {"unit": "cm", "description": "length"}


def test_unit_is_deprecated():
    t = make_tabular({"*": {"unit": "mm"}})
    with pytest.warns(FutureWarning, match="unit_of"):
        assert t.unit == "mm"
    with pytest.warns(FutureWarning):
        assert make_tabular().unit is None


def test_unit_raises_on_several_units():
    t = make_tabular({"*": {"unit": "mm"}, "std": {"unit": "cm"}})
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="several units"):
        t.unit


def test_spec_is_copied():
    spec = {"*": {"unit": "mm"}}
    t = make_tabular(spec)
    t._units.specs["*"]["unit"] = "changed"
    assert spec == {"*": {"unit": "mm"}}


def test_lazy_loader_runs_once_on_first_use():
    loader = MagicMock(return_value={"*": {"unit": "fmol/mg"}})
    t = make_tabular(loader)
    loader.assert_not_called()
    assert t.unit_of("mean") == "fmol/mg"
    assert t.unit_of("std") == "fmol/mg"
    loader.assert_called_once()


def test_data_attrs_units():
    t = make_tabular({"*": {"unit": "mm"}, "std": {"unit": "cm"}})
    data = t.data
    assert data.attrs["units"] == {"mean": "mm", "std": "cm"}  # no default for the label column
    s = data["mean"]
    assert s.attrs["units"][s.name] == "mm"
    assert make_tabular().data.attrs["units"] == {}


def test_data_attrs_row_units():
    t = make_tabular({"x": {"unit": "mm"}, "y": {"unit": "mm^2"}})
    assert t.data.attrs["units"] == {"x": "mm", "y": "mm^2"}
    assert t.data.loc["y"].attrs["units"]["y"] == "mm^2"


def test_units_table():
    t = make_tabular({"*": {"unit": "mm"}, "std": {"description": "standard deviation"}, "y": {"unit": "cm"}})
    units = t.units
    assert list(units.columns) == ["Units", "Description", "LongName"]
    assert units.loc["mean", "Units"] == "mm"
    assert units.loc["std", "Description"] == "standard deviation"
    assert units.loc["y", "Units"] == "cm"
    assert "label" not in units.index
    assert make_tabular().units.empty


def test_from_bids_sidecar():
    sidecar = {
        "thickness": {"Units": "mm", "Description": "Cortical thickness", "LongName": "Thickness", "Levels": {}},
        "Name": "not a column entry",
    }
    units = _Units.from_bids_sidecar(sidecar)
    assert units.specs == {
        "thickness": {"unit": "mm", "description": "Cortical thickness", "long_name": "Thickness"}
    }


def test_to_bids_sidecar_expands_default_to_numeric_columns():
    units = _Units({"*": {"unit": "mm"}, "std": {"description": "standard deviation"}, "x": {"unit": "cm"}})
    assert units.to_bids_sidecar(DATA) == {
        "mean": {"Units": "mm"},
        "std": {"Units": "mm", "Description": "standard deviation"},
    }  # no default for the label column, no row entries
    assert _Units().to_bids_sidecar(DATA) == {}


def test_bids_round_trip():
    sidecar = {"mean": {"Units": "mm", "Description": "average"}}
    assert _Units.from_bids_sidecar(sidecar).to_bids_sidecar(DATA) == sidecar


@pytest.mark.parametrize("unit, expected", [
    ({"*": {"unit": "mm"}}, {"mean": {"Units": "mm"}, "std": {"Units": "mm"}}),
    (None, None),
])
def test_to_zip_writes_sidecar(unit, expected):
    t = make_tabular(unit)
    buffer = BytesIO()
    with ZipFile(buffer, "w") as fh:
        t._to_zip(fh)
    with ZipFile(buffer) as fh:
        assert "tabular.csv" in fh.namelist()
        if expected is None:
            assert "tabular.json" not in fh.namelist()
        else:
            assert json.loads(fh.read("tabular.json")) == expected


def test_factory_passes_unit():
    spec = {
        "@type": "siibra/feature/tabular/v0.1",
        "file": "https://example.org/table.csv",
        "modality": "m",
        "region": "hoc1",
        "species": "Homo sapiens",
        "unit": {"*": {"unit": "mm"}, "volume": {"unit": "mm^3"}},
    }
    t = Factory.from_json(spec)
    assert t.unit_of("thickness") == "mm"
    assert t.unit_of("volume") == "mm^3"


def make_profile(values=np.array([1.0, 2.0])):
    return CorticalProfile(
        description="d", modality="m", anchor=None,
        depths=np.array([0.1, 0.5]), values=values,
        unit={"*": {"unit": "cells / 0.1mm3"}},
    )


def test_cortical_profile_unit():
    p = make_profile()
    assert list(p.data.columns) == ["m (cells / 0.1mm3)"]
    assert p.data.attrs["units"] == {"m (cells / 0.1mm3)": "cells / 0.1mm3"}
    merged = CorticalProfile._merge_elements([p, p], description="d", modality="m", anchor=None)
    assert merged.unit_of(merged.data.columns[0]) == "cells / 0.1mm3"
    assert merged.data.attrs["units"] == {"m mean (cells / 0.1mm3)": "cells / 0.1mm3", "std": "cells / 0.1mm3"}


def test_cortical_profile_merge_requires_same_units():
    other = CorticalProfile(
        description="d", modality="m", anchor=None,
        depths=np.array([0.1, 0.5]), values=np.array([1.0, 2.0]), unit={"*": {"unit": "mm"}},
    )
    with pytest.raises(AssertionError):
        CorticalProfile._merge_elements([make_profile(), other], description="d", modality="m", anchor=None)


def test_plot_ylabels_use_unit_of():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    anchor = MagicMock()
    anchor.regions = {}
    data = pd.DataFrame({"mean": [1.0, 2.0], "std": [0.1, 0.2]}, index=["x", "y"])
    t = Tabular(description="d", modality="m", anchor=anchor, data=data, unit={"*": {"unit": "mm"}})
    assert t.plot().get_ylabel() == "mean \u00b1 std\nmm"
    assert Tabular(description="d", modality="m", anchor=anchor, data=data).plot().get_ylabel() == "mean \u00b1 std"
    p = make_profile()
    p._anchor_cached = anchor
    assert p.plot().get_ylabel() == "cells / 0.1mm3"
