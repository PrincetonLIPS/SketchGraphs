Onshape setup
=============

We include a set of functions (`sketchgraphs.onshape.call`) for interacting with the CAD program `Onshape <https://www.onshape.com/>`_ in order to solve geometric constraints.
The SketchGraphs `demo notebook <https://github.com/PrincetonLIPS/SketchGraphs/blob/master/demos/sketchgraphs_demo.ipynb>`_ demonstrates their main usage.

Before calling these for the first time, a small bit of setup is required.

Account & credentials
---------------------

- Visit `Onshape <https://www.onshape.com/>`_  and create an account (it's free).
- Create an API key at https://dev-portal.onshape.com/keys. Be sure to enable read and write permissions so that we may send and retrieve CAD sketches from Onshape.
- Save the file at `sketchgraphs/onshape/creds/creds.json`


Create document
---------------
We'll need a document to serve as the target for our sketches.

- Go to https://cad.onshape.com/documents and click Create->Document in the upper left.
- The document URL will be needed to perform API calls. For example, in the `demo notebook <https://github.com/PrincetonLIPS/SketchGraphs/blob/master/demos/sketchgraphs_demo.ipynb>`_, we manually paste in the target URL.
- Now that we have a new document, we must set the version identifiers of our `feature_template.json` accordingly. Run the following whenever working with a new document (set the variable `url` accordingly):

>>> url=https://cad.onshape.com/documents/6f6d14f8facf0bba02184e88/w/66a5db71489c81f4893101ed/e/120c56983451157d26a7102d
>>> python -m sketchgraphs.onshape.call --action update_template --url $url  --enable_logging

You should see a "request succeeded" included in the log if the configuration is correct.

Our Onshape setup is complete! Please reach out with any questions.