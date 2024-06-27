# Proposal: API Changes for GPyTorch 2.0

## Philosophy
- Composition over inheritance
- Keep abstractions at a minimum
- Make it simple to instantiate and use common models (e.g. via sensible defaults).

## Goals
- Ensure models are fast on small datasets
- Keep things easily extendable
- Condense down to core functionality
- Improve the documentation, type hints and shape hints

## API Changes

### GP Models

- **Do not force the user to write a subclass just to instantiate a standard (approximate) GP.** All the user should need to pass is a prior mean, kernel and likelihood, where the mean and likelihood could even have sensible defaults. Forcing subclassing creates unnecessary boilerplate and increases the time to get an initial model up and running.

### Classes and Modules which were removed

- [ ] ``PredictionStrategy``
- Deprecated modules
    - [x] ``AbstractVariationalGP``
    - [x] ``PyroVariationalGP``

## Other Changes
- Removed [shebang lines](https://stackoverflow.com/questions/9783482/should-python-library-modules-start-with-usr-bin-env-python?rq=3) from modules that are not intended to be executed in the command line.
- [Ignored ``F722`` error by ``flake8`` to enable compatibility of forward annotation with ``jaxtyping``.](https://docs.kidger.site/jaxtyping/faq/#flake8-or-ruff-are-throwing-an-error)

## Ideas
- Separate class for ``MultiOutputGaussianProcess`` (alternatively ``VectorValuedGaussianProcess``, ``MultiTaskGaussianProcess``)
- ``model.predict[ive](test_x)`` as shorthand / replacement for ``likelihood(model(test_x))``
- ``model.prior[_predictive](test_x)`` instead of prior mode as a context manager?

## Open Questions
- How diligent should we about deprecating? Or do we just accept breaking backwards compatibility.
- How to deal with updating a model with new training data? What role does ``get_fantasy_model`` play?
- Why can train inputs be lists / tuples of tensors?
- **How can we ensure we delete the cache each training loop?** How is this done currently?

## Nice-to-have but optional
- [ ] Modernize configuration a bit
    - [ ] move setup info into ``setup.cfg`` or ``pyproject.toml``
    - [ ] move tooling settings into ``pyproject.toml``
