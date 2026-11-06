"""Unit tests for Points, backed by TSV files written to a tmp_path."""

import numpy as np
import pandas as pd
import pytest

from siibra.core.region import Region
from siibra.geometries import Points
from siibra.geometries.providers import TsvPointsProvider

LEFT = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0], "area_um2": [10.0, 20.0]})
RIGHT = pd.DataFrame({"x": [-1.0], "y": [-2.0], "z": [-3.0], "area_um2": [30.0]})
HOC1_LEFT, HOC1_RIGHT = Region("hOc1 left"), Region("hOc1 right")
HOC1 = Region("hOc1", children=[HOC1_LEFT, HOC1_RIGHT])
ROOT = Region("points test parcellation", children=[HOC1])
LINKS = {"hOc1 left": [HOC1_LEFT], "hOc1 right": [HOC1_RIGHT]}


@pytest.fixture
def sources(tmp_path):
    LEFT.to_csv(tmp_path / "left.tsv", sep="\t", index=False)
    RIGHT.to_csv(tmp_path / "right.tsv", sep="\t", index=False)
    return {"hOc1 left": str(tmp_path / "left.tsv"), "hOc1 right": str(tmp_path / "right.tsv")}


@pytest.fixture
def points(isolated_cache, sources):
    return Points({}, [TsvPointsProvider(sources)], name="cells", fragment_regions=LINKS)


def test_dataframe_with_fragment_column(points):
    table = points.fetch()
    assert list(table.columns) == ["x", "y", "z", "area_um2", "fragment"]
    assert table["fragment"].tolist() == ["hOc1 left", "hOc1 left", "hOc1 right"]
    assert list(table["fragment"].cat.categories) == ["hOc1 left", "hOc1 right"]
    np.testing.assert_array_equal(table["area_um2"], [10, 20, 30])


def test_fetch_by_region(points):
    assert points.fetch(region=HOC1_RIGHT)["area_um2"].tolist() == [30]
    assert len(points.fetch(region="hOc1")) == 3


def test_unfragmented_points_have_no_fragment_column(isolated_cache, sources):
    table = Points({}, [TsvPointsProvider(sources["hOc1 left"])]).fetch()
    assert list(table.columns) == ["x", "y", "z", "area_um2"]


def test_numpy_is_memory_mapped_for_one_file(points):
    coordinates = points.fetch(fragment="hOc1 left", format="numpy")
    assert isinstance(coordinates, np.memmap) and not coordinates.flags.writeable
    np.testing.assert_array_equal(coordinates, LEFT[["x", "y", "z"]].to_numpy())
    assert points.fetch(format="numpy").shape == (3, 3)


def test_affine_is_applied_to_both_formats(isolated_cache, sources):
    affine = np.eye(4)
    affine[:3, 3] = [100, 0, 0]
    points = Points({}, [TsvPointsProvider(sources)], affine=affine)
    assert points.fetch(fragment="hOc1 right")["x"].tolist() == [99]
    np.testing.assert_array_equal(points.fetch(fragment="hOc1 right", format="numpy"), [[99, -2, -3]])


def test_unknown_format_raises(points):
    with pytest.raises(ValueError, match="formats"):
        points.fetch(format="ply")
