from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class LazyVariableRepresentationTree(object):

    def __init__(self, lazy_var):
        self._cls = lazy_var.__class__
        self._kwargs = lazy_var._kwargs

        counter = 0
        self.children = []
        for arg in lazy_var._args:
            if hasattr(arg, "representation"):  # Is it a lazy variable?
                representation_size = len(arg.representation())
                self.children.append(
                    (
                        slice(counter, counter + representation_size, None),
                        LazyVariableRepresentationTree(arg),
                    )
                )
                counter += representation_size
            else:
                self.children.append((counter, None))
                counter += 1

    def __call__(self, *flattened_representation):
        unflattened_representation = []

        for index, subtree in self.children:
            if subtree is None:
                unflattened_representation.append(flattened_representation[index])
            else:
                sub_representation = flattened_representation[index]
                unflattened_representation.append(subtree(*sub_representation))

        return self._cls(*unflattened_representation, **self._kwargs)
