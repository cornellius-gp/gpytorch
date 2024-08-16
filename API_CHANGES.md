# Proposal: API Changes for GPyTorch 2.0

## Philosophy
- Composition over inheritance
- Keep abstractions at a minimum
- Keep things easily extendable

## Goals
- Make it simple to instantiate and use common models (e.g. via sensible defaults).
- Ensure models are fast on small datasets
- Condense down to core functionality
- Improve the documentation, type hints and shape hints

## API Changes

### GP Models

- **Do not force the user to write a subclass just to instantiate a standard (approximate) GP.** All the user should need to pass is a prior mean, kernel and likelihood, where the mean and likelihood could even have sensible defaults. Forcing subclassing creates unnecessary boilerplate and increases the time to get an initial model up and running.

### Classes and Modules which were removed

- [ ] ``PredictionStrategy``
- [ ] ``beta_features``
- Deprecated modules
    - [x] ``AbstractVariationalGP``
    - [x] ``PyroVariationalGP``

### Naming
- [ ] `gpytorch.mlls` -> `gpytorch.loss_fns/losses`

## Other Changes
- Removed [shebang lines](https://stackoverflow.com/questions/9783482/should-python-library-modules-start-with-usr-bin-env-python?rq=3) from modules that are not intended to be executed in the command line.

## Ideas
- Separate class for ``MultiOutputGaussianProcess`` (alternatively ``VectorValuedGaussianProcess``, ``MultiTaskGaussianProcess``)
- ``model.predict[ive](test_x)`` as shorthand / replacement for ``likelihood(model(test_x))``
- ``model.prior[_predictive](test_x)`` instead of prior mode as a context manager?

## Open Questions
- How diligent should we about deprecating? Or do we just accept breaking backwards compatibility.
- How to deal with updating a model with new training data? What role does ``get_fantasy_model`` play?
- Why can train inputs be lists / tuples of tensors?

### Caching

#### How it works currently
- When calling ``model.train()`` on any ``gpytorch.Module``, ``self._clear_cache`` gets called, which should be overwritten.

#### What we would like
- Cache computed quantities when predicting (eval mode) multiple times
- Cache quantities computed during training for eval mode
- Make it easier for developers to use the caching

#### Implementation Ideas
- cache object vs cached methods/properties of ``ApproximationStrategy``
- Idea (improved):
    - for every property of an ApproximationStrategy() use a @cached decorator, which registers a certain backward hook that clears the cache
        - https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution
    - What about if the training data gets modified? How do we ensure the cache is cleared?
        - Can we provide a set of arguments to the @cached decorator which specify when this cached property gets cleared (i.e. on model.train(), model.eval(), when updating the training data, on the backward pass at this node)?

## Nice-to-have but optional
- [ ] Modernize configuration a bit
    - [ ] move setup info into ``setup.cfg`` or ``pyproject.toml``
    - [ ] move tooling settings into ``pyproject.toml``
- [ ] Refactor RFFKernel / InducingPointKernel into an approximation strategy?

## TODOs
- [ ] Write a test that checks consistency between approximation strategy cache and the objects in the GP (e.g. if i update parameters in GP, the cache just points to those objects)
