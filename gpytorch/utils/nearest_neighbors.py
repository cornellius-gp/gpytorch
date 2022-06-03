import torch
import warnings

from torch.nn import Module


class NNUtil(Module):
    r"""
    Utility for nearest neighbor search. It would first try to use `faiss`_ (requiring separate pacakge installment)
    as the backend for better computational performance. Otherwise, `scikit-learn` would be used as it is pre-installed
    with gpytorch.

    :param int k: number of nearest neighbors
    :param int size: number of data points on which to perform nearest neighbor search
    :param str preferred_nnlib: currently supports `faiss` and `scikit-learn`.

    Example:
        >>> train_x = torch.randn(10, 5)
        >>> nn_util = NNUtil(k=3, size=train_x.shape[0])
        >>> nn_util.set_nn_idx(train_x)
        >>> test_x = torch.randn(2, 5)
        >>> test_nn_indices = nn_util.find_nn_idx(test_x) # finding 3 nearest neighbors for test_x
        >>> test_nn_indices = nn_util.find_nn_idx(test_x, k=2) # finding 2 nearest neighbors for test_x
        >>> sequential_nn_idx = nn_util.build_sequential_nn_idx(train_x) # build up sequential nearest neighbor structure for train_x

    .. _faiss:
    https://github.com/facebookresearch/faiss
    """

    def __init__(self, k, size, preferred_nnlib="faiss"):
        super().__init__()
        self.k = k
        self.size = size

        if preferred_nnlib == "faiss":
            try:
                import faiss
                import faiss.contrib.torch_utils
                self.nnlib = "faiss"
                self.cpu()  # Initializes the index

            except ImportError:
                warnings.warn(
                    "Tried to import faiss, but failed. Falling back to scikit-learn NN.",
                    ImportWarning
                )
                self.nnlib = "sklearn"

        else:
            self.nnlib = "sklearn"

    def cuda(self, device=None):
        super().cuda(device=device)
        if self.nnlib == "faiss":
            from faiss import StandardGpuResources, GpuIndexFlatL2
            self.res = StandardGpuResources()
            self.index = GpuIndexFlatL2(self.res, self.size)
        return self

    def cpu(self):
        super().cpu()
        if self.nnlib == "faiss":
            from faiss import IndexFlatL2
            self.res = None
            self.index = IndexFlatL2(self.size)
        return self

    def find_nn_idx(self, x, k=None):
        """
        Find :math:`k` nearest neighbors for test data `x` among the training data stored in this utility
        :param x: test data
        :param k: number of nearest neighbors. Default is the value used in utility initialization. If users specify
        a different value, it needs to be smaller than or equal to the initialized value.
        :return: the indices of nearest neighbors in the training data
        """
        if k is None:
            k = self.k
        else:
            assert k <= self.k, f"User-specified k={k} should be smaller than or equal to the initialized k={self.k}."
        with torch.no_grad():
            if self.nnlib == 'sklearn':
                # TODO: check this
                if self.neighbors is None:
                    raise RuntimeError("The nearest neighbor set has not been defined. First call `set_nn_idx`")
                nn_idx = torch.from_numpy(self.train_neighbors.kneighbors(x.cpu().numpy())[1])
                nn_idx = nn_idx.long().to(x.device)
            else:
                nn_idx = self.index.search(x, k)[1]
        return nn_idx

    def set_nn_idx(self, x):
        """
        Set the indices of training data to facilitate nearest neighbor search. If users want to perform nearest
        neighbor search on new training data, this function needs to be called again.
        :param x: training data points
        :return: None
        """
        with torch.no_grad():
            if self.nnlib == 'sklearn':
                import sklearn.neighbors as NearestNeighbors
                x = x.cpu().numpy()
                self.neighbors = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(x)
            elif self.nnlib == "faiss":
                self.index.reset()
                self.index.add(x)

    def build_sequential_nn_idx(self, x):
        r"""
        Build the sequential :math:`k` nearest neighbor structure within training data in the folloiwng way:
        for the :math:`i`-th data point :math:`x_i`, find its :math:`k` nearest neighbors among preciding
        training data :math:`x_1, \cdots, x_{i-1}`, for `i=k+1:N` where `N` is the size of training data.
        :param x: training data. Shape `(N, D)`
        :return: indices of nearest neighbors. Shape: `(N-k, k)`
        """
        assert self.nnlib == 'faiss'
        from faiss import IndexFlatL2

        # building nearest neighbor structure within inducing points
        N, D = x.shape
        with torch.no_grad():
            gpu_index = IndexFlatL2(D)
            if self.res is not None:
                from faiss import index_cpu_to_gpu
                gpu_index = index_cpu_to_gpu(self.res, 0, gpu_index)

            nn_xinduce_idx = torch.empty(N - self.k, self.k, dtype=torch.int64)

            x_np = x.data.float().cpu().numpy()
            gpu_index.add(x_np[:self.k])
            for i in range(self.k, N):
                row = x_np[i][None, :]
                nn_xinduce_idx[i-self.k].copy_(
                    torch.from_numpy(gpu_index.search(row, self.k)[1][..., 0,:]).long().to(x.device)
                )
                gpu_index.add(row)
        return nn_xinduce_idx

    def to(self, device=None, dtype=None):
        """
        Put the utility to a cpu or gpu device
        """
        if "cpu" in str(device):
            return self.cpu()
        elif "cuda" in str(device):
            return self.cuda()
        else:
            raise ValueError(f"Unknown device {device}")