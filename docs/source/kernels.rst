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

:hidden:`CosineKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CosineKernel
   :members:

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

:hidden:`AdditiveStructureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:hidden:`MultiDeviceKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiDeviceKernel
   :members:


.. autoclass:: AdditiveStructureKernel
   :members:

:hidden:`ProductKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProductKernel
   :members:

:hidden:`ProductStructureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProductStructureKernel
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

:hidden:`LCMKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LCMKernel
   :members:

:hidden:`MultitaskKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskKernel
   :members:


Kernels for Scalable GP Regression Methods
--------------------------------------------

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
