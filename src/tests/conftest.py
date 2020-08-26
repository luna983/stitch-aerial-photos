import pytest

import pathlib


@pytest.fixture
def data_dir(tmp_path):
    data_dir = pathlib.Path(__file__).parent.resolve() / 'data'
    assert data_dir.is_dir()
    return data_dir
