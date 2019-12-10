Multitask/Multioutput GPs with Exact Inference
================================================

Exact GPs can be used to model vector valued functions, or functions that represent multiple tasks.
There are several different cases:

Multi-output (vector valued functions)
----------------------------------------

- **Correlated output dimensions**: this is the most common use case.
  See the `Multitask GP Regression`_ example, which implements the inference strategy defined in `Bonilla et al., 2008`_.
- **Independent output dimensions**: here we will use an independent GP for each output.

  - If the outputs share the same kernel and mean, you can train a `Batch Independent Multioutput GP`_.
  - Otherwise, you can train a `ModelList Multioutput GP`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Multitask_GP_Regression.ipynb
   Batch_Independent_Multioutput_GP.ipynb
   ModelList_GP_Regression.ipynb

Scalar function with multiple tasks
----------------------------------------

See the `Hadamard Multitask GP Regression`_ example.
This setting should be used only when each input corresponds to a single task.

.. toctree::
   :maxdepth: 1
   :hidden:

   Hadamard_Multitask_GP_Regression.ipynb


.. _Multitask GP Regression:
  ./Multitask_GP_Regression.ipynb

.. _Bonilla et al., 2008:
  https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction

.. _Batch Independent Multioutput GP:
  ./Batch_Independent_Multioutput_GP.ipynb

.. _ModelList Multioutput GP:
  ./ModelList_GP_Regression.ipynb

.. _Hadamard Multitask GP Regression:
  ./Hadamard_Multitask_GP_Regression.ipynb
