Sketchgraph models
==================

This page describes the models implemented in sketchgraphs, as well as details their usage.
The models are based on a Graph Neural Network architecture, modelling the sketch as a graph
with vertices given by entities and edges given by their constraints.

Quickstart
===========

For an initial quick-start, we recommend users to start with the provided sequence files and
associated quantization maps, and following the default hyper-parameter values for training.
Additionally, we strongly recommend using a powerful GPU, as training is compute intensive.

For example, the generative model may be trained by running:
.. code-block: bash
    python -m sketchgraphs_models.graph.train
You may monitor the training progress on the standard output, or through `tensorboard <https://www.tensorflow.org/tensorboard>`_.

Similarly, the autoconstrain model may be trained by running:
.. code-block: bash
    python -m sketchgraphs_models.autoconstraint.train


Native extensions
=================
In order to enjoy the best training performance, we strongly recommend you compile the native extensions for
the models. They are provided as a `pytorch extension <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_,
and you will require access to a C++ compiler as well as the CUDA toolkit and compiler. If you do not have access,
the models will automatically fall back to a plain python / pytorch implementation (however, there will be a
performance penalty due to a substantial amount of looping). The extensions may be compiled by running the following
command from the root directory:
.. code-block: bash
    python setup.py build_ext --inplace

