.. role:: hidden
    :class: hidden-section

gpytorch.kernels
===================================

.. automodule:: gpytorch.kernels
.. currentmodule:: gpytorch.kernels


If you don't know what kernel to use, we recommend that you start out with a
:code:`gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel)`.


Kernel
----------------

.. autoclass:: Kernel
   :members:

Standard Kernels
-----------------------------

:hidden:`LinearKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearKernel
   :members:

:hidden:`MaternKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaternKernel
   :members:

:hidden:`PeriodicKernel`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PeriodicKernel
   :members:

:hidden:`RBFKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernel
   :members:

:hidden:`SpectralMixtureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralMixtureKernel
   :members:

:hidden:`WhiteNoiseKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: WhiteNoiseKernel
   :members:


Composition/Decoration Kernels
-----------------------------------

:hidden:`AdditiveKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveKernel
   :members:

:hidden:`ProductKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProductKernel
   :members:

:hidden:`ScaleKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ScaleKernel
   :members:



Specialty Kernels
-----------------------------------

:hidden:`IndexKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: IndexKernel
   :members:

:hidden:`MultitaskKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskKernel
   :members:


Kernels for Scalable GP Regression Methods
--------------------------------------------

:hidden:`AdditiveGridInterpolationKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveGridInterpolationKernel
   :members:

:hidden:`GridKernel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridKernel
   :members:

:hidden:`GridInterpolationKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridInterpolationKernel
   :members:

:hidden:`InducingPointKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InducingPointKernel
   :members:

:hidden:`MultiplicativeGridInterpolationKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveGridInterpolationKernel
   :members:
