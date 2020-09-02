.. SketchGraphs documentation master file, created by
   sphinx-quickstart on Sun Jul 12 05:33:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SketchGraphs' documentation!
========================================

`SketchGraphs <https://github.com/PrincetonLIPS/SketchGraphs>`_ is a dataset of 15 million sketches extracted from real world CAD models intended to facilitate
research in ML-aided design and geometric program induction.
In addition to the raw data, we provide several processed datasets, an accompanying Python package to work with
the data, as well as a couple of starter models which implement GNN-type strategies on a couple of example problems.


Data
----

We provide our dataset in a number of forms, some of which may be more appropriate for your desired usage.
The following data files are provided:

- The raw json data as obtained from Onshape. This is provided as a set of 128 tar archives which are compressed
  using zstandard_. They total about 43GB of data. In general, we only recommend these for advanced usage, as
  they are heavy and require extensive processing to manipulate. Users interested in working with the raw data
  may wish to inspect `sketchgraphs.pipeline.make_sketch_dataset` and `sketchgraphs.pipeline.make_sequence_dataset`
  to view our data processing scripts. The data is available for download `here <https://sketchgraphs.cs.princeton.edu/shards/>`_.

- A dataset of construction sequences. This is provided as a single file, stored in a custom binary format.
  This format is much more concise (as it eliminates many of the unique identifiers used in the raw JSON format),
  and is better suited for ML applications. It is supported by our python libraries, and forms the baseline
  on which our models are trained. The data is available for download `here <https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy>`_
  (warning: 15GB file!).

- A filtered dataset of construction sequences. This is provided as a single file, similarly stored in a custom
  binary format. This dataset is similar to the sequence dataset, but simplified by filtering out sketches
  that are too large or too small, and only includes a simplified set of entities and constraints (while still
  capturing a large portion of the data). Additionally, this dataset has been split into training, testing and
  validation splits for convenience. We train our models on this subset of the data. You can find download the splits
  here: `train <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy>`_,
  `test <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_test.npy>`_,
  `validation <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.npy>`_.


More details concerning the data can be found in the :doc:`data` page.


.. _zstandard: https://facebook.github.io/zstd/


Models
------

In addition to the dataset, we also provide some baseline model implementations to tackle the tasks of generative
modelling and autoconstrain. These models are based on Graph Neural Network approaches, and model the sketch as
a graph where vertices are given by the entities in the sketch, and edges by the constraints between those entities.
For more details, please refer to the dedicated :doc:`models` page.


.. toctree::
   models
   data
   onshape_setup
.. autosummary::
   :toctree: _autosummary
   :recursive:

   sketchgraphs
   sketchgraphs_models



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
