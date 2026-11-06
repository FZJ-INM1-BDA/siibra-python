import pytest

from siibra.retrieval.cache import CACHE


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Keep conversions written during a test out of the user's cache."""
    folder = tmp_path / "cache"
    folder.mkdir()
    monkeypatch.setattr(CACHE, "folder", str(folder))
    return folder
