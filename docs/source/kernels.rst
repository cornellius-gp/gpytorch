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


:hidden:`CylindricalKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CylindricalKernel
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

:hidden:`PolynomialKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialKernel
   :members:

:hidden:`PolynomialKernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialKernelGrad
   :members:

:hidden:`RBFKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernel
   :members:

:hidden:`RQKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RQKernel
   :members:

:hidden:`SpectralDeltaKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralDeltaKernel
   :members:

:hidden:`SpectralMixtureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralMixtureKernel
   :members:


Composition/Decoration Kernels
-----------------------------------

:hidden:`AdditiveKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveKernel
   :members:

:hidden:`MultiDeviceKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiDeviceKernel
   :members:

:hidden:`AdditiveStructureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

:hidden:`ArcKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ArcKernel
   :members:

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

:hidden:`RBFKernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernelGrad
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

:hidden:`RFFKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RFFKernel
   :members:
