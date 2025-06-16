.. BioCRNPyler documentation master file, created by
   sphinx-quickstart on Thu Jan 31 19:56:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AutoReduce's documentation!
====================================

AutoReduce is a Python package for automated model reduction of SBML models. It provides tools for:

* Automated model reduction using QSSA (Quasi-Steady State Approximation)
* Hill function approximation
* Integration with BioCRNPyler for synthetic biology models
* Analysis of gene expression models

Installation
-----------

You can install AutoReduce using pip:

.. code-block:: bash

    pip install autoreduce

For development installation with all optional dependencies:

.. code-block:: bash

    pip install -e ".[all]"

Quick Start
----------

Here's a simple example of using AutoReduce to reduce a model using conservation laws and timescale separation:

.. code-block:: python

    from autoreduce.converters import load_sbml

    # Load your SBML model
    sys = load_sbml('your_sbml_file.xml', outputs=['your_output'])

    # Solve conservation laws
    conservation_laws = sys.solve_conservation_laws(
        conserved_sets=[
            ['species1', 'species2', 'species3'],  # First conserved set
            ['species4', 'species5']               # Second conserved set
        ],
        states_to_eliminate=['species_to_eliminate1', 'species_to_eliminate2']
    )

    # Solve timescale separation using QSSA
    reduced_qssa_model = sys.solve_timescale_separation(['fast_species1', 'fast_species2'])

For more detailed examples, see the :doc:`examples` section.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
