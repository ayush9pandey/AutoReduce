Contributing
============

We welcome contributions to AutoReduce. See :doc:`develop` for the full
developer notes and release checklist.

Development Setup
-----------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/<your-username>/AutoReduce.git
       cd AutoReduce

3. Create a new branch:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

4. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

Code Style
----------

We use ruff for linting and formatting. Configuration is defined in
``pyproject.toml``.

.. code-block:: bash

    ruff check autoreduce tests
    ruff format autoreduce tests

Running Tests
-------------

Run the test suite with:

.. code-block:: bash

    pytest

For coverage report:

.. code-block:: bash

    pytest --cov=autoreduce --cov-report=xml

Documentation
-------------

The documentation is built using Sphinx. To build it locally:

1. Install documentation dependencies:

   .. code-block:: bash

       pip install -e ".[docs]"

2. Build the docs:

   .. code-block:: bash

       cd docs
       make html

3. View the documentation by opening ``docs/_build/html/index.html``

Pull Request Process
--------------------

1. Update the documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

License
-------

By contributing to AutoReduce, you agree that your contributions will be licensed under the project's BSD License.
