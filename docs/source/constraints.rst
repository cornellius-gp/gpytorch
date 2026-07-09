.. role:: hidden
    :class: hidden-section

gpytorch.constraints
===================================

.. automodule:: gpytorch.constraints
.. currentmodule:: gpytorch.constraints


Parameter Constraints
-----------------------------

Constraints keep a parameter's value within a valid domain (for example, strictly positive or
bounded in an interval) while the optimizer works on an unconstrained tensor under the hood.
GPyTorch stores the unconstrained value of a parameter; the constraint is applied on every read
so the model only ever sees a valid value.

The base class is :class:`Interval` (a closed interval ``[lower_bound, upper_bound]``), and the
most common derived classes are:

* :class:`Positive` -- ``x > 0`` (the most common constraint, used for lengthscales, noise
  scales, and outputscales).
* :class:`GreaterThan` -- ``x >= lower_bound`` (a one-sided lower bound).
* :class:`LessThan` -- ``x <= upper_bound`` (a one-sided upper bound).

A constraint is attached to a parameter via
:meth:`~gpytorch.module.Module.register_constraint`, which adds a property of the same name to
the module that returns the constrained value of the parameter.

Example
~~~~~~~~~~~~~~~~~~~~~~~~~

The example below constrains a custom parameter to be strictly positive and shows the round-trip
behavior of the underlying transform.

.. code-block:: python

   import torch
   import gpytorch

   class MyModule(gpytorch.Module):
       def __init__(self):
           super().__init__()
           self.register_parameter(
               "raw_scale", torch.nn.Parameter(torch.tensor(0.0)),
           )
           # Expose ``self.scale`` as a strictly positive quantity.
           self.register_constraint("raw_scale", gpytorch.constraints.Positive())

       @property
       def scale(self):
           # Auto-generated property: returns the constrained value.
           return self._constraints["raw_scale_constraint"].transform(self.raw_scale)

   m = MyModule()
   m.scale  # -> tensor constrained to (0, infinity)


Constraint Reference
-----------------------------

:hidden:`Interval`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Interval
   :members:

:hidden:`GreaterThan`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GreaterThan
   :members:

:hidden:`Positive`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Positive
   :members:

:hidden:`LessThan`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LessThan
   :members:
