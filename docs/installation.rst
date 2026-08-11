Installation
============

Requirements
------------

Supported Python versions are: 3.9 - 3.13.

Required runtime dependencies are listed below
but will be automatically installed with pip when you install the package:

* python-libsbml
* sympy
* scipy
* numpy

Optional compatibility and visualization dependencies:

* matplotlib
* seaborn
* biocrnpyler, for BioCRNpyler model construction workflows
* control, for python-control `NonlinearIOSystem` adapters
* pydmd, for DMD and DMDc projection workflows

Basic Installation
------------------

You can install AutoReduce using pip:

.. code-block:: bash

    pip install autoreduce

To install AutoReduce with all optional compatibility packages, use:

.. code-block:: bash

    pip install "autoreduce[all]"

Or install from source:

.. code-block:: bash

    git clone https://github.com/ayush9pandey/AutoReduce.git
    cd AutoReduce
    pip install .

Development Installation
------------------------

For development, install the package in editable mode with the dependencies
needed for the task:

.. code-block:: bash

    git clone https://github.com/ayush9pandey/AutoReduce.git
    cd AutoReduce
    pip install -e ".[dev]"

Use the feature extras explicitly when working on optional integrations:

.. code-block:: bash

    pip install -e ".[all]"
    pip install -e ".[bio]"
    pip install -e ".[control]"
    pip install -e ".[dmd]"

Verifying Installation
-----------------------

To verify your installation, you can run Python and import the package:

.. code-block:: python

    import autoreduce
    print(autoreduce.__version__)

If you don't see any errors, the installation was successful.
