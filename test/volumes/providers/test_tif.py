"""Tests for the 2D tif volume provider."""

import numpy as np
import pytest
import tifffile

from siibra.locations.boundingbox import BoundingBox
from siibra.retrieval import requests
from siibra.volumes.providers.nifti import Tif2DNiftiProvider
from siibra.volumes.volume import Volume

# 0.5 mm pixels, 2 mm thick slice, offset (10, 20, 30) mm
AFFINE = np.array([
    [0.5, 0.0, 0.0, 10.0],
    [0.0, 0.5, 0.0, 20.0],
    [0.0, 0.0, 2.0, 30.0],
    [0.0, 0.0, 0.0, 1.0],
])
MAXPOINT = (10.0 + 0.5 * 6, 20.0 + 0.5 * 4, 30.0 + 2.0)


@pytest.fixture
def image():
    # 4 rows (y) x 6 cols (x); every value encodes its position
    rows, cols = np.mgrid[0:4, 0:6]
    return (10 * rows + cols).astype("uint16")


def make_provider(tmp_path, arr):
    path = tmp_path / "section.tif"
    tifffile.imwrite(path, arr)
    return Tif2DNiftiProvider({"url": str(path), "transform": AFFINE.tolist()})


@pytest.fixture
def provider(tmp_path, image):
    return make_provider(tmp_path, image)


@pytest.mark.parametrize("suffix", [".tif", ".tiff"])
def test_decoder_is_registered(suffix):
    url = f"https://example.org/section{suffix}"
    assert requests.Decoder.from_filename(url) is requests.DECODERS[suffix]


def test_fetch_places_columns_along_x_and_rows_along_y(provider, image):
    img = provider.fetch()
    assert img.shape == (6, 4, 1)
    assert np.allclose(img.affine, AFFINE)
    assert np.array_equal(np.asarray(img.dataobj)[..., 0], image.T)


def test_fetch_label_returns_mask(provider, image):
    label = image[2, 3]
    mask = provider.fetch(label=label)
    assert np.array_equal(np.asarray(mask.dataobj)[..., 0], (image == label).T)


def test_boundingbox(provider):
    bbox = provider.get_boundingbox()
    assert np.allclose(bbox.minpoint.coordinate, (10.0, 20.0, 30.0))
    assert np.allclose(bbox.maxpoint.coordinate, MAXPOINT)


def test_boundingbox_rejects_unsupported_kwargs(provider):
    with pytest.raises(NotImplementedError):
        provider.get_boundingbox(voi=BoundingBox((0, 0, 0), (1, 1, 1), space=None))


def test_rgb_image_is_rejected(tmp_path):
    provider = make_provider(tmp_path, np.zeros((4, 6, 3), dtype="uint8"))
    with pytest.raises(ValueError, match="RGB"):
        provider.fetch()


def test_volume_uses_the_provider(provider, image):
    volume = Volume(space_spec={}, providers=[provider], name="tif")
    assert volume.fetch(label=image[1, 1]) is not None
    bbox = volume.get_boundingbox(clip=False)
    assert np.allclose(bbox.maxpoint.coordinate, MAXPOINT)
