.. role:: hidden
    :class: hidden-section

gpytorch.nearest_neighbors
===================================

These modules provide a set of interfaces for partitioning datasets and establishing
neighborhood structures between partitions. This kind of partitioning is required for
nearest-neighbor-style Gaussian Process models, and we ensure behind the scenes that nearest-neighbor models
based on these partitions still form valid joint density functions.

.. automodule:: gpytorch.nearest_neighbors
.. currentmodule:: gpytorch.nearest_neighbors


Indexes
-----------------------------

Indexes are the interfaces used to partition datasets with clustering algorithms, measure distance
between partitions with a distance metric for establishing neighboring structure, and ordering
the data with ordering strategies.

:hidden:`KMeansIndex`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KMeansIndex
   :members:


:hidden:`VoronoiIndex`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VoronoiIndex
   :members:


Distance Metrics
-----------------------------

Distance metrics are used to define distances between partitions of data. Each index defines the
points that represent each block, and distance between blocks is defined as the distance between
these representatives per the supplied distance metric. The DistanceMetrics class includes methods
for Euclidean distance and Manhattan distance metrics, and custom distance metrics must return
functions that take in vectors of observations and return the distance matrix for those observations.

:hidden:`DistanceMetrics`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DistanceMetrics
   :members:


Ordering Strategies
-----------------------------

Because nearest-neighbor approximations depend on the ordering of the data they're trained on, we need a way
to order the dataset by different metrics to find the best ordering strategy for a given problem.
The OrderingStrategies class includes methods for ordering the data by a given coordinate or by an
:math:`L_p` norm. Custom ordering strategies can be implemented here and must return a function that
takes in a vector of observations and returns a vector of integers indicating the index of each observation
under the new ordering.

:hidden:`OrderingStrategies`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OrderingStrategies
   :members:
