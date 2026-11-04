import pytest

from backend import config


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own model cache so nothing leaks between tests or into cache/."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
