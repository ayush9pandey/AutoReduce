Installation
============

Requirements
-----------

AutoReduce requires Python 3.9 or higher and the following dependencies:

* python-libsbml
* sympy
* scipy
* numpy

Optional dependencies (for visualization and advanced features):

* matplotlib
* seaborn

Basic Installation
----------------

You can install AutoReduce using pip:

.. code-block:: bash

    pip install autoreduce

Or install from source:

.. code-block:: bash

    git clone https://github.com/yourusername/autoreduce.git
    cd autoreduce
    pip install .

Development Installation
----------------------

For development, you can install the package in editable mode with all optional dependencies:

.. code-block:: bash

    git clone https://github.com/ayush9pandey/AutoReduce.git
    cd AutoReduce
    pip install -e ".[all]"

This will install the package in development mode, allowing you to modify the code and see changes immediately.

Verifying Installation
--------------------

To verify your installation, you can run Python and import the package:

.. code-block:: python

    import autoreduce
    print(autoreduce.__version__)

If you don't see any errors, the installation was successful.

Troubleshooting
--------------

If you encounter any issues during installation:

1. Make sure you have Python 3.9 or higher installed
2. Try creating a fresh virtual environment
3. Check that all dependencies are properly installed
4. If using conda, you might need to install some packages through conda instead of pip

For more help, please open an issue on the `GitHub repository <https://github.com/ayush9pandey/AutoReduce/issues>`_.
