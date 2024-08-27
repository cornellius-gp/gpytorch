# Proposal: API Changes for GPyTorch 2.0

## Philosophy
- Composition over inheritance
- Keep abstractions at a minimum
- Keep things easily extendable

## Goals
- Ensure models are fast on small datasets
- Condense down to core functionality
- Improve the documentation, type hints and shape hints
- Make it simple to instantiate and use common models (e.g. via sensible defaults).

## API Changes

### GP Models

### Classes and Modules which were removed

- [ ] ``PredictionStrategy``
- [ ] ``beta_features``
- Deprecated modules
    - [x] ``AbstractVariationalGP``
    - [x] ``PyroVariationalGP``

### Naming
- [ ] `gpytorch.mlls` -> `gpytorch.loss_fns/losses`
- [ ] `ExactMarginalLogLikelihood` -> `LogMarginalLikelihood`

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
- Should training data be saved with the model (i.e. be a persistent or non-persistent buffer?)

### Caching

#### How it works currently
- When calling ``model.train()`` on any ``gpytorch.Module``, ``self._clear_cache`` gets called, which should be overwritten.
- context manager to determine whether caches should remain on computation graph
- variational strategy: flag for Pytorch buffers: https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/variational/_variational_strategy.py

#### What we would like
- Cache computed quantities when predicting (eval mode) multiple times
- Cache quantities computed during training for reuse (e.g. in explicit backward pass)
- Cache quantities computed during training for eval mode
- Make it easier for developers to use the caching

#### Implementation Ideas
- Discussion with Geoff:
    - be explicit when buffers are released / populated (rather than doing it implicitly like memoization or cached_property)
    - be offensive in releasing to avoid hard to debug situation
    - designs
        1. Flag for buffers, which we check everytime we would use a buffer, and if necessary overwrite buffer. Pytorch buffers are always detached from the graph.
        2. Add-on: backward hook, add data hook which automatically release the buffer
        3. .register_cache[d_quantity/tensor]() pattern like .register_parameter() / .register_prior() in ``gpytorch.Module`` (see also https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/module.py#L203), could even have an argument on what hook is registered for that buffer.

## Nice-to-have but optional
- [ ] Go through all the `DeprecationWarning`s and remove obsolete code.
- [ ] Modernize configuration a bit
    - [ ] move setup info into ``setup.cfg`` or ``pyproject.toml``
    - [ ] move tooling settings into ``pyproject.toml``
- [ ] Refactor RFFKernel / InducingPointKernel into an approximation strategy?

## TODOs
- [ ] Enable Multi-task models for Gaussian Process
- [ ] Ensure `gpytorch.settings.fast_pred_var(True)` still works
- [ ] Add `TestGaussianProcess` test case
    - [ ] Move tests in `TestCholeskyGP` to appropriate files
