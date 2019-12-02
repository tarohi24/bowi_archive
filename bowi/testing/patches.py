import pytest

from bowi import settings


@pytest.fixture(autouse=True)
def patch_data_dir(monkeypatch):
    monkeypatch.setattr(settings, 'data_dir',
                        settings.project_root.joinpath('bowi/tests/data'))
