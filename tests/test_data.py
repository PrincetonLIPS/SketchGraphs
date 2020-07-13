#pylint: disable=missing-module-docstring,missing-function-docstring

import json
import os
import tarfile

import numpy as np

from sketchgraphs.data.sequence import sketch_to_sequence, NodeOp, EdgeOp
from sketchgraphs.data.sketch import Sketch, render_sketch, EntityType, ConstraintType, SubnodeType, get_sequence_dof


def sketch_json_sample():
    sample_file = 'tests/testdata/sample_json.tar.xz'

    result = []

    with tarfile.open(sample_file, 'r:xz') as tar_archive:
        for f in tar_archive:
            if not f.isfile():
                continue
            result.extend(json.load(tar_archive.extractfile(f)))

    return result


def test_sketch_from_json():
    sketch_json_list = sketch_json_sample()

    for sketch_json in sketch_json_list:
        Sketch.from_fs_json(sketch_json)


def test_sequence_from_sketch():
    sketch_json_list = sketch_json_sample()

    for sketch_json in sketch_json_list:
        sketch = Sketch.from_fs_json(sketch_json)
        seq = sketch_to_sequence(sketch)

def test_plot_sketch():
    sketch_json_list = sketch_json_sample()[:10]

    for sketch_json in sketch_json_list:
        fig = render_sketch(Sketch.from_fs_json(sketch_json))
        assert fig is not None


def test_get_sequence_dof():
    seq = [
        NodeOp(label=EntityType.External),
        NodeOp(label=EntityType.Line),
        NodeOp(label=SubnodeType.SN_Start),
        EdgeOp(label=ConstraintType.Subnode, references=(2, 1)),
        NodeOp(label=SubnodeType.SN_End),
        EdgeOp(label=ConstraintType.Subnode, references=(3, 1)),
        NodeOp(label=EntityType.Line),
        EdgeOp(label=ConstraintType.Parallel, references=(4, 1)),
        EdgeOp(label=ConstraintType.Horizontal, references=(4,)),
        EdgeOp(label=ConstraintType.Distance, references=(4, 1)),
        NodeOp(label=SubnodeType.SN_Start),
        EdgeOp(label=ConstraintType.Subnode, references=(5, 4)),
        NodeOp(label=SubnodeType.SN_End),
        EdgeOp(label=ConstraintType.Subnode, references=(6, 4)),
        NodeOp(label=EntityType.Stop)]

    dof_remaining = np.sum(get_sequence_dof(seq))
    assert dof_remaining == 5
