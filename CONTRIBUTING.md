# Contributing to GPyTorch

Thanks for contributing!

## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/cornellius-gp/gpytorch.git
cd gpytorch
pip install -e .[dev,docs,examples,keops,pyro,test]  # keops and pyro are optional
pre-commit install
```


## Our Development Process

### Formatting and Linting

GPyTorch uses [pre-commit](https://pre-commit.com) for code formatting
and [flake8](https://flake8.pycqa.org/en/latest/) for linting.
This enforces a common code style across the repository.
The [development installation instructions](#development-installation) should install both tools, and no additional configuration should be necessary.

`flake8` and `pre-commit` are both run every time you make a local commit.
To run both commands independent of making a commit:
```bash
SKIP=flake8 pre-commit run --files test/**/*.py gpytorch/**/*.py
flake8
```

### Docstrings
We use [standard sphinx docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) (not Google-style).


### Type Hints

GPyTorch aims to be fully typed using Python 3.10+
[type hints](https://www.python.org/dev/peps/pep-0484/).

We recognize that we have a long way to go towards fully typing the library,
but we expect all contributions to also use proper type annotations.
While we currently do not enforce full consistency of these in our continuous integration
test, you should strive to type check your code locally.
For this we recommend using [pyre](https://pyre-check.org/).


### Unit Tests

We use python's `unittest` to run unit tests:
```bash
python -m unittest
```

- To run tests within a specific directory, run (e.g.) `python -m unittest discover -s test/kernels`.
- To run a specific unit test, run (e.g.) `python -m unittest test.operators.test_rbf_kernel.TestRBFKernel.test_active_dims_range`.


### Documentation

GPyTorch uses sphinx to generate documentation, and ReadTheDocs to host documentation.
To build the documentation locally, ensure that sphinx and its plugins are properly installed (see the [development installation section](#development-installation) for instructions).
Then run:

```bash
cd docs
SKIP_EXAMPLES=true make html
cd build/html
python -m http.server 8000
```

If you are making an update to one of the example notebooks, drop the `SKIP_EXAMPLES=true` environment variable:
(This takes longer to build.)

```bash
cd docs
make html
cd build/html
python -m http.server 8000
```

The documentation will be available at http://localhost:8000.
You will have to rerun the `make html` or `SKIP_EXAMPLES=true make html` command every time you wish to update the docs.

**Important**: Our CI requires that the documentation compiles *WITHOUT WARNINGS*.
Please ensure that you can build the documentation locally without producing any warnings.

## Pull Requests
We greatly appreciate PRs! To minimze back-and-forward communication, please ensure that your PR includes the following:

1. **Code changes.** (the bug fix/new feature/updated documentation/etc.)
1. **Unit tests.** If you are updating any code, you should add an appropraite unit test.
   - If you are fixing a bug, make sure that there's a new unit test that catches the bug.
     (I.e., there should be a new unit test that fails before your bug fix, but passes after your bug fix.
     This ensures that we don't have any accidental regressions in the code base.)
   - If you are adding a new feature, you should add unit tests for this new feature.
1. **Documentation.** Any new objects/methods should have [appropriate docstrings](#docstrings).
   - If you are adding a new object, **please ensure that it appears in the documentation.**
     You may have to add the object to the appropriate file in [docs/source](https://github.com/cornellius-gp/gpytorch/tree/main/docs/source).
1. **Example notebooks.** Any major new functionality, tutorials, or examples should have an example jupyter notebook.
   - If you are adding a new notebook, **please ensure that it appears in the documentation.**
     You may have to add the object to the appropriate file in [docs/source](https://github.com/cornellius-gp/gpytorch/tree/main/examples/).

Before submitting a PR, ensure the following:
1. **Code is proprerly formatted and linted.** Linting and formatting checking should happen automatically if you have followed the development installation instructions.
   See [the formatting and linting](#formatting-and-linting) section for more info.
1. **Unit tests pass.** See [the unit tests section](#unit-tests) for more info.
1. **Documentation renders correctly without warnings.** [Build the documentation locally](#documentation) to ensure that your new class/docstrings are rendered correctly. Ensure that sphinx can build the documentation without warnings.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

We accept the following types of issues:
- Bug reports
- Requests for documentation/examples
- Feature requests
- Opportuntities to refactor code
- Performance issues (speed, memory, etc.)

Please refrain from using the issue tracker for questions or debugging personal code.
Instead please use the [GPyTorch discussions forum](https://github.com/cornellius-gp/gpytorch/discussions).

## License

By contributing to GPyTorch, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
