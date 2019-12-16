PyTorch NN Integration (Deep Kernel Learning)
===============================================

Because GPyTorch is built on top of PyTorch, you can seamlessly integrate existing PyTorch modules into GPyTorch models.
This makes it possible to combine neural networks with GPs, either with exact or approximate inference.

Here we provide some examples of **Deep Kernel Learning**, which are GP models that use kernels parameterized by neural networks.

- **Exact inference GP + NN**: see the `Exact DKL`_ (*deep kernel learning*) example, based on `Wilson et al., 2015`_.
- **Approximate inference GP + NN**: see the `CIFAR-10 Classification SVDKL`_ (*stochastic variational deep kernel learning*) example, based on `Wilson et al., 2016`_.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   KISSGP_Deep_Kernel_Regression_CUDA.ipynb
   Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb

.. _Exact DKL:
  KISSGP_Deep_Kernel_Regression_CUDA.ipynb

.. _CIFAR-10 Classification SVDKL:
  Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb

.. _Wilson et al., 2015:
  https://arxiv.org/abs/1511.02222

.. _Wilson et al., 2016:
  https://arxiv.org/abs/1611.00336
