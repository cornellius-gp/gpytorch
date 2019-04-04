#!/usr/bin/env python3

import warnings
import torch
from copy import deepcopy
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .. import settings
from .gp import GP
from .exact_prediction_strategies import prediction_strategy


class ExactGP(GP):
    def __init__(self, train_inputs, train_targets, likelihood):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    @property
    def train_targets(self):
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value):
        object.__setattr__(self, '_train_targets', value)

    def _apply(self, fn):
        if self.train_inputs is not None:
            self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
            self.train_targets = fn(self.train_targets)
        return super(ExactGP, self)._apply(fn)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        Args:
            - :attr:`inputs` the new training inputs
            - :attr:`targets` the new training targets
            - :attr:`strict`
                if `True`, the new inputs and targets must have the same shape, dtype, and device
                as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
        small, because any computed test-time caches will be updated in linear time rather than computed from scratch.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.

        Args:
            - :attr:`inputs` (Tensor `m x d` or `b x m x d`): Locations of fantasy observations.
            - :attr:`targets` (Tensor `m` or `b x m`): Labels of fantasy observations.
        Returns:
            - :class:`ExactGP`
                An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
                and all test-time caches have been updated.
        """
        if self.prediction_strategy is None:
            raise RuntimeError("Fantasy observations can only be added after making predictions with a model so that "
                               "all test independent caches exist. Call the model on some data first!")

        if self.train_inputs[0].dim() > 2:
            raise RuntimeError("Adding fantasy observations to a GP that is already batch mode will not be supported "
                               "until GPyTorch supports multiple batch dimensions.")

        if self.train_targets.dim() > 1:
            raise RuntimeError("Cannot yet add fantasy observations to multitask GPs, but this is coming soon!")

        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = list(i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs)

        # If input is n x d but targets is b x n x d, expand input to b x n x d
        for i, input in enumerate(inputs):
            if input.dim() == targets.dim():
                inputs[i] = input.unsqueeze(0).repeat(*targets.shape[:-1], 1, 1)
                batch_shape = inputs[i].shape[:-2]
            elif input.dim() == targets.dim() + 1:
                batch_shape = input.shape[:-2]
            elif input.dim() < targets.dim():
                batch_shape = targets.shape[:-2]
                inputs[i] = input.expand(*batch_shape, *input.shape[-2:])

        train_inputs = [tin.expand(*batch_shape, *tin.shape[-2:]) for tin in list(self.train_inputs)]
        train_targets = self.train_targets.expand(*batch_shape, *self.train_targets.shape[-2:])

        full_inputs = [torch.cat([train_input, input], dim=-2) for train_input, input in zip(train_inputs, inputs)]
        full_targets = torch.cat([train_targets, targets], dim=-1)

        full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)

        # Copy model without copying training data or prediction strategy (since we'll overwrite those)
        old_pred_strat = self.prediction_strategy
        old_train_inputs = self.train_inputs
        old_train_targets = self.train_targets
        self.prediction_strategy = None
        self.train_inputs = None
        self.train_targets = None
        new_model = deepcopy(self)
        self.prediction_strategy = old_pred_strat
        self.train_inputs = old_train_inputs
        self.train_targets = old_train_targets

        new_model.train_inputs = full_inputs
        new_model.train_targets = full_targets
        new_model.prediction_strategy = old_pred_strat.get_fantasy_strategy(
            inputs,
            targets,
            full_inputs,
            full_targets,
            full_output
        )

        return new_model

    def train(self, mode=True):
        if mode:
            self.prediction_strategy = None
        return super(ExactGP, self).train(mode)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.prediction_strategy = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = list(i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args)

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.check_training_data.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = super(ExactGP, self).__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output
        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?", UserWarning
                    )

            # Exact inference
            non_batch_train = False

            if all(tin.dim() == 2 for tin in self.train_inputs):
                # Train inputs were non-batch
                non_batch_train = True
            # Expand train_input to match input's batch size, or vice versa if they don't match.
            for i, (train_input, input) in enumerate(zip(train_inputs, inputs)):
                if train_input.dim() < input.dim():
                    batch_shape = inputs[0].shape[:-2]
                    train_inputs[i] = train_input.expand(*batch_shape, *train_input.shape[-2:])
                elif input.dim() < train_input.dim():
                    batch_shape = train_inputs[0].shape[:-2]
                    inputs[i] = input.expand(*batch_shape, *input.shape[-2:])
                else:
                    batch_shape = train_inputs[0].shape[:-2]

            full_inputs = tuple(
                torch.cat([train_input, input], dim=-2) for train_input, input in zip(train_inputs, inputs)
            )

            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            train_targets = self.train_targets
            if any(
                orig_train_input.dim() < train_input.dim()
                for orig_train_input, train_input in zip(self.train_inputs, train_inputs)
            ):
                train_targets = train_targets.unsqueeze(0).expand(*batch_shape, *train_targets.size())

            num_tasks = 1
            if isinstance(full_output, MultitaskMultivariateNormal):
                num_tasks = full_output.num_tasks

            full_mean = full_mean.view(*batch_shape, -1)
            num_train = train_targets.size(len(batch_shape))
            train_targets = train_targets.view(*batch_shape, -1)

            train_mean = full_mean.narrow(-1, 0, train_targets.size(-1))

            if self.prediction_strategy is None:
                train_train_covar = full_covar[..., :num_train, :num_train]
                self.prediction_strategy = prediction_strategy(
                    num_train,
                    train_inputs,
                    train_mean,
                    train_train_covar,
                    train_targets,
                    self.likelihood,
                    non_batch_train,
                )

            if train_inputs[0].shape != self.prediction_strategy.train_inputs[0].shape:
                # The test batch shape has changed, update prediction strategy with expanded objects
                batch_shape = train_inputs[0].shape[:-2]
                train_train_covar = self.prediction_strategy.train_train_covar
                if len(batch_shape) > len(train_train_covar.batch_shape):
                    # Expanding to add more batches
                    expanded_covar = train_train_covar.expand(*batch_shape, *train_train_covar.matrix_shape)
                elif (
                    len(batch_shape) == len(train_train_covar.batch_shape)
                    and batch_shape.numel() != train_train_covar.batch_shape.numel()
                ):
                    # The test batch size has changed, we need to repeat it to a new batch size.
                    expanded_covar = train_train_covar[0].expand(*batch_shape, *train_train_covar.matrix_shape)
                else:
                    # We are leaving batch mode, not entering it.
                    expanded_covar = train_train_covar[0]
                self.prediction_strategy.train_train_covar = expanded_covar
                self.prediction_strategy.num_train = num_train
                self.prediction_strategy.train_targets = train_targets
                self.prediction_strategy.train_inputs = train_inputs
                self.prediction_strategy.train_mean = train_mean
                self.prediction_strategy.non_batch_train = non_batch_train

            test_mean = full_mean.narrow(-1, train_targets.size(-1), full_mean.size(-1) - train_targets.size(-1))
            if full_mean.size(-1) <= settings.max_eager_kernel_size.value() and num_tasks == 1:
                test_covar = full_covar[..., num_train:, :].evaluate_kernel()
                test_test_covar = test_covar[..., num_train:]
                test_train_covar = test_covar[..., :num_train]
            else:
                test_test_covar = full_covar[..., num_train:, num_train:]
                test_train_covar = full_covar[..., num_train:, :num_train]

            with settings._use_eval_tolerance():
                predictive_mean = self.prediction_strategy.exact_predictive_mean(test_mean, test_train_covar)
                predictive_covar = self.prediction_strategy.exact_predictive_covar(test_test_covar, test_train_covar)

            if num_tasks > 1:
                if train_targets.ndimension() == 2:
                    # Batch multitask
                    predictive_mean = predictive_mean.view(train_targets.size(0), -1, num_tasks).contiguous()
                else:
                    # Standard multitask
                    predictive_mean = predictive_mean.view(-1, num_tasks).contiguous()

            return full_output.__class__(predictive_mean, predictive_covar)
