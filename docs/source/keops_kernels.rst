.. role:: hidden
    :class: hidden-section

gpytorch.kernels.keops
===================================

.. automodule:: gpytorch.kernels.keops
.. currentmodule:: gpytorch.kernels.keops


These kernels are compatible with the GPyTorch KeOps integration.
For more information, see the `KeOps tutorial`_.

.. note::
   Only some standard kernels have KeOps impementations.
   If there is a kernel you want that's missing, consider submitting a pull request!


.. _KeOps Tutorial:
   examples/02_Scalable_Exact_GPs/KeOps_GP_Regression.html


:hidden:`RBFKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernel
   :members:


:hidden:`MaternKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaternKernel
   :members:


:hidden:`PeriodicKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PeriodicKernel
   :members:
