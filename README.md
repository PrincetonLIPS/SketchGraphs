# SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry in Computer-Aided Design

SketchGraphs is a dataset of 15 million sketches extracted from real-world CAD models intended to facilitate research in both ML-aided design and geometric program induction.

![blah](/assets/sketchgraphs.gif)


Each sketch is represented as a geometric constraint graph where edges denote designer-imposed geometric relationships between primitives, the nodes of the graph.

![Sketch and graph](/assets/sketch_w_graph.png)

Video: https://youtu.be/ki784S3wjqw  
Paper: https://arxiv.org/abs/2007.08506 

See [demo notebook](demos/sketchgraphs_demo.ipynb) for a quick overview of the data representions in SketchGraphs as well as an example of solving constraints via Onshape's API.

## Installation 

SketchGraphs can be installed using pip: 

```bash
>> pip install -e SketchGraphs 
```

This will provide you with the necessary dependencies to load and explore the data.
However, to train the models, you will need to additionally install [pytorch](https://pytorch.org/)
and [torch-scatter](https://github.com/rusty1s/pytorch_scatter).

## Data

We provide our dataset in a number of forms, some of which may be more appropriate for your desired usage.
The following data files are provided:

- The raw json data as obtained from Onshape. This is provided as a set of 128 tar archives which are compressed
  using [zstandard](https://facebook.github.io/zstd). They total about 43GB of data. In general, we only recommend these for advanced usage, as
  they are heavy and require extensive processing to manipulate. Users interested in working with the raw data
  may wish to inspect `sketchgraphs.pipeline.make_sketch_dataset` and `sketchgraphs.pipeline.make_sequence_dataset`
  to view our data processing scripts. The data is available for download [here](https://sketchgraphs.cs.princeton.edu/shards).

- A dataset of construction sequences. This is provided as a single file, stored in a custom binary format.
  This format is much more concise (as it eliminates many of the unique identifiers used in the raw JSON format),
  and is better suited for ML applications. It is supported by our Python libraries and forms the baseline
  on which our models are trained. The data is available for download [here](https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy) (warning: 15GB file!).

- A filtered dataset of construction sequences. This is provided as a single file, similarly stored in a custom
  binary format. This dataset is similar to the sequence dataset, but simplified by filtering out sketches
  that are too large or too small, and only includes a simplified set of entities and constraints (while still
  capturing a large portion of the data). Additionally, this dataset has been split into training, testing and
  validation splits for convenience. We train our models on this subset of the data. You can download the splits
  here: [train](https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy)
  [validation](https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.npy)
  [test](https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_test.npy)

For full documentation of the processing pipeline, see https://princetonlips.github.io/SketchGraphs.

The original creators of the CAD sketches hold the copyright. See [Onshape Terms of Use 1.g.ii](https://www.onshape.com/legal/terms-of-use#your_content) for additional licensing details.


## Models
In addition to the dataset, we also provide some baseline model implementations to tackle the tasks of generative
modeling and autoconstrain. These models are based on Graph Neural Network approaches and model the sketch as
a graph where vertices are given by the entities in the sketch, and edges by the constraints between those entities.
For more details, please refer to https://princetonlips.github.io/SketchGraphs/models.


## Citation
If you use this dataset in your research, please cite:
```
@inproceedings{SketchGraphs,
  title={Sketch{G}raphs: A Large-Scale Dataset for Modeling Relational Geometry in Computer-Aided Design},
  author={Seff, Ari and Ovadia, Yaniv and Zhou, Wenda and Adams, Ryan P.},
  booktitle={ICML 2020 Workshop on Object-Oriented Learning},
  year={2020}
}
```
