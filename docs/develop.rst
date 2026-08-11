.. currentmodule:: autoreduce

***************
Developer Notes
***************

This chapter contains notes for developers contributing to AutoReduce.

Package Structure
=================

The package source is organized by model-reduction responsibility:

- `autoreduce.system` contains the core `System` representation and model
  import adapters.
- `autoreduce.solvers` contains numerical ODE and sensitivity solvers.
- `autoreduce.reductions` contains time-scale, conservation, abundance, and
  projection-based reduction methods.
- `autoreduce.utils` contains conversion, SBML, and reduction-constructor
  utilities.
- `examples` is split into biological, canonical, and ecological examples.
- `tests` contains the pytest suite.

Code Style
==========

AutoReduce uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and
formatting configuration. Run these commands from the repository root:

.. code-block:: bash

   ruff check autoreduce tests
   ruff format autoreduce tests

Docstrings should follow NumPy style. Public modules, classes, and functions
should explain the scientific object being represented, the assumptions made,
and the expected symbolic or numeric dimensions.

Testing
=======

Install the test dependencies and run pytest:

.. code-block:: bash

   pip install -e ".[dev]"
   pytest

Run coverage explicitly when needed:

.. code-block:: bash

   pytest --cov=autoreduce --cov-report=xml

Documentation
=============

Documentation is built with Sphinx from the ``docs`` directory. Install the
documentation extra from the repository root:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Build the HTML documentation locally:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

On Windows, if a synchronized folder or a browser locks files under
``docs/_build``, build to a fresh output directory instead:

.. code-block:: bash

   python -m sphinx -b html docs %TEMP%\autoreduce-sphinx-html

Read the Docs
-------------

AutoReduce is configured for Read the Docs with ``.readthedocs.yaml`` at the
repository root. The Read the Docs project should:

1. Import the GitHub repository.
2. Use the repository configuration file, ``.readthedocs.yaml``.
3. Build with the Sphinx configuration in ``docs/conf.py``.
4. Install the package with the ``docs`` extra.

The current Read the Docs configuration uses Python 3.12. Reproduce any Read
the Docs failure locally by running:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html docs docs/_build/html

Releasing New Versions
======================

AutoReduce uses semantic versioning (`MAJOR.MINOR.PATCH`) and Git tags through
`setuptools_scm`. Package wheels and source distributions are published to
PyPI by the manual GitHub Actions release workflow.

PyPI Setup
----------

The PyPI project should use trusted publishing from GitHub Actions. In the
PyPI project settings for ``autoreduce``, add a trusted publisher with:

- Owner: ``ayush9pandey``
- Repository: ``AutoReduce``
- Workflow: ``pypi-deploy.yml``
- Environment: ``Release Deploy``

In GitHub, create the ``Release Deploy`` environment so that it matches the
workflow environment in ``.github/workflows/pypi-deploy.yml``. Add reviewer
protection there if releases should require manual approval.

Release checklist:

1. Ensure the working tree is clean and CI is green.
2. Update documentation and examples as needed.
3. Run the package checks from the release environment:

   .. code-block:: bash

      python -m pip install -e ".[dev]"
      pytest
      ruff check autoreduce tests
      python -m sphinx -b html docs docs/_build/html
      python -m build --no-isolation

4. Commit and push all release changes.
5. Create an annotated tag on the release commit:

   .. code-block:: bash

      git tag -a vX.Y.Z -m "Release X.Y.Z"
      git push --tags

6. Run the PyPI release workflow from the tag or provide the tag in the
   workflow `ref` input.
7. Verify the published package from a clean environment:

   .. code-block:: bash

      python -m pip install autoreduce==X.Y.Z
      python -c "import autoreduce; print(autoreduce.__version__)"

Generated build outputs such as ``build/``, ``dist/``, ``*.egg-info``, and
``autoreduce/_version.py`` are setuptools artifacts. They must not be committed.
