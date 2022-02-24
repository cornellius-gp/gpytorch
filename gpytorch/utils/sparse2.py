import torch


class SparseTensor(object):
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values

    def matmul(self, dense):
        """
        sparse represents a (n, n) matrix
        dense is of size (n, m)

        indices is of size (n, k)
        values is of size (n, k)
        """

        """
        dense[indices] returns a (n, k, m) dense matrix
        """
        result = torch.sum(self.values.unsqueeze(-1) * dense[self.indices], dim=1)
        return result


if __name__ == "__main__":
    indices = torch.tensor([[0, 2], [0, 1], [1, 2]], dtype=torch.long)
    values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)

    sparse = SparseTensor(indices, values)

    dense = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float)

    result = sparse.matmul(dense)
    print(result)

    """
    Ground truth
    [[4, 10],
     [3, 9],
     [5, 11]]
    """
