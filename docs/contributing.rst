Contributing
============

We welcome contributions to AutoReduce! This document provides guidelines and instructions for contributing.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/<your-username>/autoreduce.git
       cd autoreduce

3. Create a new branch:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

4. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[all]"
       pip install pytest pytest-cov nbval

Code Style
---------

We use flake8 for code style checking. The configuration is in ``pyproject.toml``. Key points:

* Maximum line length: 80 characters
* Follow PEP 8 guidelines

Running Tests
------------

Run the test suite with:

.. code-block:: bash

    pytest

For coverage report:

.. code-block:: bash

    pytest --cov=autoreduce

Documentation
------------

The documentation is built using Sphinx. To build it locally:

1. Install documentation dependencies:

   .. code-block:: bash

       pip install sphinx sphinx_rtd_theme nbsphinx myst_parser

2. Build the docs:

   .. code-block:: bash

       cd docs
       make html

3. View the documentation by opening ``docs/_build/html/index.html``

Pull Request Process
------------------

1. Update the documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update the changelog
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

License
-------

By contributing to AutoReduce, you agree that your contributions will be licensed under the project's BSD License. 