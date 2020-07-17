"""This module contains supporting code to the main sketchgraphs dataset. In particular, it contains two models based
on graph representations of the data, which are geared towards the task of generation (see `sketchgraphs_models.graph`)
and autoconstrain (see `sketchgraphs_models.autoconstraint`). In addition, this module contains a number of submodules
to help the implementation of these models.

These models are tuned and operate on a subset of the full sketchgraphs dataset. In particular, they only handle
constraints which relate at most two entities (e.g. the mirror constraint is not handled), and are trained on a subset
of sketches which excludes sketches that are too large or too small. In addition, these model use a quantization
strategy to model continuous parameters in the dataset, which must be pre-computed before training. To create
quantization maps, see `sketchgraphs.pipeline.make_quantization_statistics`.

"""
