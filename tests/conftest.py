"""Shared test fixtures in order to easily provide testing data to all tests."""

import json
import tarfile

import pytest

from sketchgraphs.data.sketch import Sketch


@pytest.fixture
def sketches_json():
    sample_file = 'tests/testdata/sample_json.tar.xz'

    result = []

    with tarfile.open(sample_file, 'r:xz') as tar_archive:
        for f in tar_archive:
            if not f.isfile():
                continue
            result.extend(json.load(tar_archive.extractfile(f)))

    return result

@pytest.fixture
def sketches(sketches_json):
    """Return a list of sample sketches."""
    return [Sketch.from_fs_json(j) for j in sketches_json]
