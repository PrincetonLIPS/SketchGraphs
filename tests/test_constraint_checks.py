"""Tests for constraint checking"""

import itertools

from sketchgraphs.data import constraint_checks, sketch_to_sequence, EdgeOp

def test_constraint_check(sketches):
    for sketch_idx, sketch in enumerate(itertools.islice(sketches, 1000)):
        sequence = sketch_to_sequence(sketch)
        for op_idx, op in enumerate(sequence):
            if not isinstance(op, EdgeOp):
                continue
            try:
                check_value = constraint_checks.check_edge_satisfied(sequence, op)
            except ValueError:
                check_value = True

            if check_value is None:
                check_value = True

            assert check_value, "constraint check failed at sketch {0}, op {1}".format(sketch_idx, op_idx)
