import json

import pytest

from siibra.retrieval.bids import (
    find_bids_sidecars,
    load_bids_metadata,
    parse_bids_filename,
)
from siibra.retrieval.repositories import LocalFileRepository


@pytest.mark.parametrize("name, entities, suffix, extension, datatype", [
    ("sub-01/anat/sub-01_T1w.nii.gz", {"sub": "01"}, "T1w", ".nii.gz", "anat"),
    ("sub-01_ses-pre_run-01_bold.dtseries.nii", {"sub": "01", "ses": "pre", "run": "01"}, "bold", ".dtseries.nii", None),
    ("tpl-MNI152NLin2009cAsym_atlas-Julich_dseg.tsv", {"tpl": "MNI152NLin2009cAsym", "atlas": "Julich"}, "dseg", ".tsv", None),
    ("sub-01_acq-a+b_T1w.nii", {"sub": "01", "acq": "a+b"}, "T1w", ".nii", None),
    ("participants.tsv", {}, "participants", ".tsv", None),
])
def test_parse_valid(name, entities, suffix, extension, datatype):
    for strict in (False, True):
        info = parse_bids_filename(name, strict=strict)
        assert info.entities == entities
        assert info.suffix == suffix
        assert info.extension == extension
        assert info.datatype == datatype


@pytest.mark.parametrize("name, entities, suffix", [
    ("sub-01_desc-julich-brain_T1w.nii", {"sub": "01", "desc": "julich-brain"}, "T1w"),  # kept
    ("sub-01_extra_T1w.nii", {"sub": "01"}, "T1w"),          # non key-value part skipped
    ("sub-01_sub-02_T1w.nii", {"sub": "01"}, "T1w"),         # first occurrence wins
    ("sub-01_-x_T1w.nii", {"sub": "01"}, "T1w"),             # empty key skipped
    ("sub-01_desc-x.nii", {"sub": "01", "desc": "x"}, None),  # no suffix
    ("sub-01_T1w+.nii", {"sub": "01"}, "T1w+"),              # suffix kept
])
def test_parse_lenient_and_strict(name, entities, suffix, caplog):
    with caplog.at_level("DEBUG", logger="siibra"):
        info = parse_bids_filename(name)
    assert (info.entities, info.suffix) == (entities, suffix)
    assert name in caplog.text  # every deviation is logged
    with pytest.raises(ValueError):
        parse_bids_filename(name, strict=True)


@pytest.fixture
def dataset(tmp_path):
    files = {
        "dataset_description.json": {"Name": "test", "BIDSVersion": "1.10.0"},
        "T1w.json": {"MagneticFieldStrength": 3, "Manufacturer": "A", "Nested": {"a": 1}},
        "sub-01/sub-01_T1w.json": {"Manufacturer": "B", "Nested": {"b": 2}},
        "sub-01/anat/sub-01_acq-hi_T1w.json": {"EchoTime": 0.003},
        "sub-01/anat/sub-01_acq-hi_T1w.nii.gz": None,
        "sub-01/anat/sub-01_acq-lo_T1w.nii.gz": None,
        "sub-01/anat/sub-01_bold.json": {"RepetitionTime": 2},
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"" if content is None else json.dumps(content).encode())
    return LocalFileRepository(str(tmp_path))


def test_find_sidecars_ordered_from_root(dataset):
    assert find_bids_sidecars("sub-01/anat/sub-01_acq-hi_T1w.nii.gz", dataset) == [
        "T1w.json",
        "sub-01/sub-01_T1w.json",
        "sub-01/anat/sub-01_acq-hi_T1w.json",
    ]
    assert find_bids_sidecars("sub-01/anat/sub-01_acq-lo_T1w.nii.gz", dataset) == [
        "T1w.json",
        "sub-01/sub-01_T1w.json",
    ]


def test_metadata_merged_top_down(dataset):
    assert load_bids_metadata("sub-01/anat/sub-01_acq-hi_T1w.nii.gz", dataset) == {
        "MagneticFieldStrength": 3,  # inherited from the dataset root
        "Manufacturer": "B",         # overridden at subject level
        "Nested": {"b": 2},          # replaced, not combined
        "EchoTime": 0.003,           # most specific level
    }


def test_no_sidecar(tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}")
    (tmp_path / "sub-01_dwi.nii.gz").write_bytes(b"")
    repo = LocalFileRepository(str(tmp_path))
    assert find_bids_sidecars("sub-01_dwi.nii.gz", repo) == []
    assert load_bids_metadata("sub-01_dwi.nii.gz", repo) == {}


def test_lookup_stops_at_dataset_root(tmp_path):
    (tmp_path / "T1w.json").write_text('{"Outside": true}')
    (tmp_path / "ds" / "sub-01").mkdir(parents=True)
    (tmp_path / "ds" / "dataset_description.json").write_text("{}")
    (tmp_path / "ds" / "sub-01" / "sub-01_T1w.nii").write_bytes(b"")
    repo = LocalFileRepository(str(tmp_path))
    assert find_bids_sidecars("ds/sub-01/sub-01_T1w.nii", repo) == []


def test_sidecar_lookup_requires_suffix(dataset):
    with pytest.raises(ValueError, match="no BIDS suffix"):
        find_bids_sidecars("sub-01/anat/sub-01_desc-x.nii.gz", dataset)
