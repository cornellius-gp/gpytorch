# Overview of Examples

This `examples` directory provides numerous ipython notebooks that demonstrate the use of GPyTorch.

1. [Getting started](#getting-started)
1. [Specialty Models/Tasks](#specialty-models-and-tasks)
1. [Scalable GP Regression Models](#scalable-gp-regression-models)
1. [Scalable GP Classification Models](#scalable-gp-classification-models)
1. [Deep Kernel Learning](#deep-kernel-learning)

## Getting started

These are no-frills GP models, which will work in most small data applications.
If you are looking to get familiar with GPyTorch, start here.

- **Regression** - check out the [simple regression example](01_Simple_GP_Regression/Simple_GP_Regression.ipynb)
- **Classification** - check out the [simple classification example](02_Simple_GP_Classification/Simple_GP_Classification.ipynb)

Some advanced techniques that you can apply to soup up these simple models:

- **GPU Acceleration** - see [how to use CUDA with GPyTorch](01_Simple_GP_Regression/Simple_GP_Regression_CUDA.ipynb)
- **Fast Predictive Variances w/ LOVE** - see [how to get really fast predictions with LOVE](01_Simple_GP_Regression/Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb)


## Specialty Models and Tasks

- **Multitask GP Regression** - check out the examples in the [multitask GP folder](03_Multitask_GP_Regression)
- **Bayesian Optimization** - example coming soon!


## Scalable GP Regression Models

If you have more than ~1,000 training data points, the simple GP models might start acting a bit slow.
There are multiple methods to scale up GP regression, and the correct choice depends on your application.
GPyTorch supports the following inducing point methods:
- **KISS-GP Regression** - [more info](https://arxiv.org/abs/1503.01057)
    - A [simple KISS-GP example](04_Scalable_GP_Regression_1D/KISSGP_Regression_1D.ipynb) for one-dimensional data
    - [An example](05_Scalable_GP_Regression_Multidimensional/KISSGP_Kronecker_Regression.ipynb) for low-dimensional data
    - An [example that combines KISS-GP with Deep Kernel Learning](05_Scalable_GP_Regression_Multidimensional/KISSGP_Deep_Kernel_Regression_CUDA.ipynb)
    - And [more](05_Scalable_GP_Regression_Multidimensional)!
- **SGPR** - [more info](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)
    - Example coming soon!

While there are lots of different choices, switching between methods requires a quick one-line change to your model.
In addition, it is fairly straightforward to create your own custom scalable GP method. (Tutorial coming soon!)
This is especially useful if your data is structured (e.g. if your data lies on a regularly-spaced grid).

Additionally, it is possible to use stochastic variational inference for regression problems.
This is useful if you have an extremely large dataset.
Some examples:
- A [1D example combining KISS-GP and stochastic variational inference](04_Scalable_GP_Regression_1D/KISSGP_Regression_1D_With_Stochastic_Variational_Inference_CUDA.ipynb)
- An [example combining KISS-GP, Deep Kernel Learning, and stochastic variational inference](05_Scalable_GP_Regression_Multidimensional/KISSGP_Deep_Kernel_Regression_With_Stochastic_Variational_Inference_CUDA.ipynb)


## Scalable GP Classification Models

There are multiple methods for scalable GP classification, and the correct choice depends on your application.
Some examples:
- **KISS-GP Classification**
    - A [simple KISSGP example](06_Scalable_GP_Classification_1D/KISSGP_Classification_1D.ipynb) for one-dimensional data
    - [An example](07_Scalable_GP_Classification_Multidimensional/KISSGP_Kronecker_Classification.ipynb) for low-dimensional data
    - And [more](07_Scalable_GP_Classification_Multidimensional)!


## Deep Kernel Learning

GPyTorch seemlessly integrates with PyTorch, making it extremely easy to combine GPs with neural networks.
The following examples use **[Deep Kernel Learning](https://arxiv.org/abs/1511.02222)**:
- A [large-scale regression problem](05_Scalable_GP_Regression_Multidimensional/KISSGP_Deep_Kernel_Regression_CUDA.ipynb) with Deep Kernel Learning
- Training [a GP for CIFAR image classification](08_Deep_Kernel_Learning/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb) with Deep Kernel Learning
- And [more](08_Deep_Kernel_Learning)!
