# Intro

This `examples` directory provides numerous ipython notebooks that demonstrate the use of GPyTorch.

# Which notebooks to read

If you're just starting work with Gaussian processes, check out the simple [regression](simple_gp_regression.ipynb) and
[classification](simple_gp_classification.ipynb). These show the most basic usage of GPyTorch and provide links to
useful reading material.

If you have a specific task in mind, then check out this [flowchart](flowchart.pdf) to find which notebook will help you.

Here's a verbal summary of the flow chart:

## Regression

*Do you have lots of data?*

**No:** Start with the [basic example](simple_gp_regression.ipynb)

*Is your training data one-dimensional?*

**Yes:** Use [KissGP regression](kissgp_gp_regression.ipynb)

*Does your output decompose additively?*

**Yes:** Use [Additive Grid Interpolation](kissgp_additive_regression_cuda.ipynb)

*Is your trainng data three-dimensional or less?*

**Yes**: Exploit [Kronecker structure](kissgp_kronecker_product_regression.ipynb)

**No**: Try Deep Kernel regression (example pending)

### Variational Regression (new!)

Try this if:
- You have too much data for exact inference, even with KissGP/Deep kernel learning/etc.
- Your model will need variational inference anyways (e.g. if you're doing some sort of clustering)

## Classification

*Do you have lots of data?*

**No:** Start with the [basic example](simple_gp_classification.ipynb)

*Is your training data one-dimensional?*

**Yes:** Use [KissGP classification](kissgp_gp_classification.ipynb)

*Does your output decompose additively?*

**Yes:** Use [Additive Grid Interpolation](kissgp_additive_classification_cuda.ipynb)

*Is your training data three-dimensional or less?*

**Yes**: Exploit [Kronecker structure](kissgp_kronecker_product_classification.ipynb)

**No**: Try Deep Kernel [classification]dkl_mnist.ipynb)
