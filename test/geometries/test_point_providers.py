"""Unit tests for the point table providers, backed by files written to a tmp_path."""

import os

import numpy as np
import pandas as pd
import pytest

from siibra.geometries.providers import CsvPointsProvider, NpyPointsProvider, TsvPointsProvider

LEFT = pd.DataFrame({
    "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "z": [7.0, 8.0, 9.0],
    "area_um2": [10.5, 11.0, 12.25], "layer": ["II", "III", "IV"],
})
RIGHT = pd.DataFrame({"X": [-1.0], "Y": [-2.0], "Z": [-3.0], "area_um2": [9.0], "layer": ["V"]})


@pytest.fixture
def tsv_sources(tmp_path):
    left, right = tmp_path / "hOc1_left.tsv", tmp_path / "hOc1_right.tsv"
    LEFT.to_csv(left, sep="\t", index=False)
    RIGHT.to_csv(right, sep="\t", index=False)
    return {"hOc1 left": str(left), "hOc1 right": str(right)}


def test_fragments_are_the_source_names(tsv_sources):
    assert TsvPointsProvider(tsv_sources).fragments == ["hOc1 left", "hOc1 right"]


def test_single_source_has_no_fragments(isolated_cache, tsv_sources):
    provider = TsvPointsProvider(tsv_sources["hOc1 left"])
    assert provider.fragments == []
    assert list(provider.fetch()) == [None]


def test_coordinates_are_memory_mapped(isolated_cache, tsv_sources):
    coordinates = TsvPointsProvider(tsv_sources).get_coordinates(["hOc1 left"])
    assert isinstance(coordinates, np.memmap)
    np.testing.assert_array_equal(coordinates, LEFT[["x", "y", "z"]].to_numpy())


def test_coordinate_columns_are_case_insensitive(isolated_cache, tsv_sources):
    np.testing.assert_array_equal(
        TsvPointsProvider(tsv_sources).get_coordinates(["hOc1 right"]), [[-1, -2, -3]]
    )


def test_coordinates_of_several_fragments_are_concatenated(isolated_cache, tsv_sources):
    assert TsvPointsProvider(tsv_sources).get_coordinates().shape == (4, 3)


def test_attributes_keep_names_and_types(isolated_cache, tsv_sources):
    _, attributes = TsvPointsProvider(tsv_sources).fetch(["hOc1 left"])["hOc1 left"]
    assert isinstance(attributes, np.memmap)
    assert attributes.dtype.names == ("area_um2", "layer")
    np.testing.assert_array_equal(attributes["area_um2"], LEFT["area_um2"])
    assert attributes["layer"].tolist() == ["II", "III", "IV"]


def test_tables_are_parsed_once(isolated_cache, tsv_sources, monkeypatch):
    TsvPointsProvider(tsv_sources).fetch()
    files = sorted(os.listdir(isolated_cache))

    def fail(self, path):
        raise AssertionError("parsed again")

    monkeypatch.setattr(TsvPointsProvider, "_read", fail)
    TsvPointsProvider(tsv_sources).fetch()
    assert sorted(os.listdir(isolated_cache)) == files
    assert len([f for f in files if f.endswith(".npy")]) == 4  # coordinates and attributes per table


def test_a_missing_cache_file_is_restored_from_one_parse(isolated_cache, tsv_sources, monkeypatch):
    provider = TsvPointsProvider(tsv_sources["hOc1 left"])
    provider.fetch()
    for f in os.listdir(isolated_cache):
        os.remove(os.path.join(isolated_cache, f))
    calls = []
    original = TsvPointsProvider._read
    monkeypatch.setattr(TsvPointsProvider, "_read", lambda self, path: calls.append(path) or original(self, path))
    provider.fetch()
    assert len(calls) == 1


def test_tables_without_attributes(isolated_cache, tmp_path):
    path = tmp_path / "points.tsv"
    LEFT[["x", "y", "z"]].to_csv(path, sep="\t", index=False)
    _, attributes = TsvPointsProvider(str(path)).fetch()[None]
    assert attributes is None


def test_empty_table(isolated_cache, tmp_path):
    path = tmp_path / "points.tsv"
    LEFT.iloc[:0].to_csv(path, sep="\t", index=False)
    assert TsvPointsProvider(str(path)).get_coordinates().shape == (0, 3)


def test_missing_coordinate_columns_raise(isolated_cache, tmp_path):
    path = tmp_path / "points.tsv"
    LEFT.drop(columns="z").to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="no column"):
        TsvPointsProvider(str(path)).get_coordinates()


def test_unknown_fragment_raises(isolated_cache, tsv_sources):
    with pytest.raises(ValueError, match="No fragment"):
        TsvPointsProvider(tsv_sources).fetch(["hOc2 left"])


def test_csv(isolated_cache, tmp_path):
    path = tmp_path / "points.csv"
    LEFT.to_csv(path, index=False)
    np.testing.assert_array_equal(
        CsvPointsProvider(str(path)).get_coordinates(), LEFT[["x", "y", "z"]].to_numpy()
    )


def test_npy_plain_array(isolated_cache, tmp_path):
    path = tmp_path / "points.npy"
    np.save(path, LEFT[["x", "y", "z"]].to_numpy())
    coordinates, attributes = NpyPointsProvider(str(path)).fetch()[None]
    np.testing.assert_array_equal(coordinates, LEFT[["x", "y", "z"]].to_numpy())
    assert attributes is None


def test_npy_structured_array(isolated_cache, tmp_path):
    path = tmp_path / "points.npy"
    np.save(path, LEFT[["x", "y", "z", "area_um2"]].to_records(index=False))
    coordinates, attributes = NpyPointsProvider(str(path)).fetch()[None]
    np.testing.assert_array_equal(coordinates, LEFT[["x", "y", "z"]].to_numpy())
    assert attributes.dtype.names == ("area_um2",)


def test_npy_of_other_shape_raises(isolated_cache, tmp_path):
    path = tmp_path / "points.npy"
    np.save(path, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="expected"):
        NpyPointsProvider(str(path)).get_coordinates()
