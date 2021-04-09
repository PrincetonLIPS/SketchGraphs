SketchGraphs models
===================

This page describes the models implemented in SketchGraphs, as well as details their usage.
The models are based on a Graph Neural Network architecture, modelling the sketch as a graph
with vertices given by entities and edges given by their constraints.

Quickstart
----------

For an initial quick-start, we recommend users to start with the provided sequence files and
associated quantization maps, and following the default hyper-parameter values for training.
Additionally, we strongly recommend using a powerful GPU, as training is compute intensive.

For example, assuming that you have downloaded the training dataset, as well as the accompanying
quantization statistics (available `here <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.stats.pkl.gz>`_),
the generative model may be trained by running:

.. code-block:: bash

    python -m sketchgraphs_models.graph.train --dataset_train sg_t16_train.npy

You may monitor the training progress on the standard output, or through `tensorboard <https://www.tensorflow.org/tensorboard>`_.

Similarly, the autoconstrain model may be trained by running:

.. code-block:: bash

    python -m sketchgraphs_models.autoconstraint.train --dataset_train sg_t16_train.npy

We will also provide pre-trained models (coming soon!).


Torch scatter
-------------

In order to enjoy the best training performance, we strongly recommend you install the `torch-scatter <https://github.com/rusty1s/pytorch_scatter>`_
package with the correct CUDA version for training on GPU. If you do not have access,
the models will automatically fall back to a plain python / pytorch implementation (however, there will be a
performance penalty due to a substantial amount of looping).
