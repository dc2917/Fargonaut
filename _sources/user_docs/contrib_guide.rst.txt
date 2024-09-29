Contributing Guidelines
=======================

Install the development requirements::

  $ pip install -r requirements-dev.txt

Install the pre-commit hooks::

  $ pre-commit install

Run the tests::

  $ python -m unittest discover tests

Check test coverage::

  $ coverage run --omit=tests/* -m unittest discover tests
  $ coverage report -m

Build the documentation::

  $ cd docs
  $ make html
