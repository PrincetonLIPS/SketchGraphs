Datasets
========

We provide three datasets at various levels of pre-processing in order to enable a wide variety of uses
while ensuring that it is possible to quickly get started on this data. We describe each dataset below.

Tarballs
--------

The raw json data as obtained from Onshape. This is provided as a set of 128 tar archives which are compressed
using `zstandard <https://facebook.github.io/zstd/>`_.
They total about 43GB of data. In general, we only recommend these for advanced usage, as
they are heavy and require extensive processing to manipulate.
Users who wish to directly access the json from python may be interested in the utility function
`sketchgraphs.pipeline.make_sketch_dataset.load_json_tarball`, which iterates through a single compressed tarball
and enumerates the sketches in the tarball.
The data is available for download `here <https://sketchgraphs.cs.princeton.edu/shards/>`_.


Filtered sequence
-----------------

The filtered sequences are the most convenient dataset to use, and are provided ready to use with ML models.
They are stored in a custom binary format, which can be read using `sketchgraphs.data.flat_array.load_dictionary_flat`,
by for example executing:

>>> from sketchgraphs.data import flat_array
>>> data = flat_array.load_dictionary_flat('sg_t16_train.npy')
>>> data['sequences']
FlatArray(len=9179789, mem=5.3 GB)

In addition to the sequences, which may be accessed through the ``data['sequences']`` key, the sequence files
also contain an integer array of the same length which record the length of each sequence, accessed through
the ``data['sequence_lengths']`` key, and a structured array which contains an identifier uniquely identifying
the sketch, accessed through the ``data['sketch_ids']`` key. The latter is an array of tuples, the first element
being the document id of the sketch, the second being the part index in that document, and the last being the
sketch index in that part.

This data is pre-split into a `train <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy>`_,
`test <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_test.npy>`_ and
`validation <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.npy>`_ set (the training set
is formed from the shards 1-120, the validation set from the shards 121-124, and the testing set from the
shards 125-128). Additionally, we also provide pre-computed `quantization statistics <https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.stats.pkl.gz>`_,
computed on the training set using the `sketchgraphs.pipeline.make_quantization_statistics` script.
These quantization statistics are used by the models in order to handle continuous parameters.


Full sequence
-------------

As a middle-ground between the filtered sequences and the json tarballs, we also provide the full sequences,
which are directly converted from the json tarballs with no filtering (except the exclusing of empty sketches).
This full sequence file is available `here <https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy>`_
(warning: 15GB download), and contains the sequence representation of all the sketches in the dataset.
We note that although it is mostly equivalent to the data contained in the tarballs for ML purposes, it is much smaller
as many of the original identifiers are discarded and renamed to sequential indices.

The full sequence may be accessed in the same fashion as the filtered sequences (using the `sketchgraphs.data.flat_array`
module). Note that the full sequence file contains all sketches, and does not provide any train / test split.
Additionally, users should be warned that the dataset contains substantial outliers (for example, the largest sketch in the
dataset contains more than 350 thousand operations, whereas the 99th percentile is "only" 640 operations).

