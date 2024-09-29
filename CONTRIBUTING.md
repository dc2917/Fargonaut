# Contributing to Fargonaut

Contributions to Fargonaut are welcome.

If you have a suggestion, please open an issue describing in detail the change you would like to be made. 

Similarly, if you have found a bug or a mistake, please open an issue describing this in detail.

If you would like to contribute code, please:

1. Open an issue to detail the change/addition you wish to make, unless one already exists

1. Fork the repository

1. Make a clone of your fork

1. Install the development requirements

   ```bash
   $ python -m unittest discover tests
   ```

1. Install the pre-commit hooks:

   ```bash
   $ python -m unittest discover tests
   ```

1. Create a feature branch and make your changes

1. Ensure that all tests pass

   * Run the tests via

   ```bash
   $ python -m unittest discover tests
   ```

   * Check the test coverage via

   ```bash
   $ coverage run --omit=tests/* -m unittest discover tests
   $ coverage report -m
   ```

1. Ensure that the documentation builds successfully

   * Build the documentation via

   ```bash
   $ cd docs
   $ make html
   ```

1. Open a pull request. In the pull request description, please describe in detail the changes your feature branch introduces, and reference the associated issue.
