import pytest

from bowi import settings


@pytest.fixture
def datadir_patch(mocker):
    mocker.patch.object(settings,
                        'data_dir',
                        settings.project_root.joinpath('tests/data'))
