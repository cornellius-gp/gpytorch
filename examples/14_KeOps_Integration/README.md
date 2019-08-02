KeOps (https://github.com/getkeops/keops) is a recently released software package for fast kernel operations that integrates wih PyTorch. We can use the ability of KeOps to perform efficient kernel matrix multiplies on the GPU to integrate with the rest of GPyTorch.

In this folder, we'll demonstrate how to integrate the kernel matmuls of KeOps with all of the bells of whistles of GPyTorch, including things like our preconditioning for conjugate gradients.

In our example notebook, we will train an exact GP on `3droad`, which has hundreds of thousands of data points. Together, the highly optimized matmuls of KeOps combined with algorithmic speed improvements like preconditioning allow us to train on a dataset like this in a matter of minutes using only a single GPU.