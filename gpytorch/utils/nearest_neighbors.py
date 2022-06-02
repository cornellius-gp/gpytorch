import torch
import warnings
import time

from .. import Module


class NNUtil(Module):
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
        if k is None:
            k = self.k
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
        with torch.no_grad():
            if self.nnlib == 'sklearn':
                import sklearn.neighbors as NearestNeighbors
                x = x.cpu().numpy()
                self.neighbors = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(x)
            elif self.nnlib == "faiss":
                self.index.reset()
                self.index.add(x)

    def build_idx_for_inducing_points(self, inducing_points, M, D):
        assert self.nnlib == 'faiss'
        from faiss import IndexFlatL2

        # building nearest neighbor structure within inducing points
        with torch.no_grad():
            gpu_index = IndexFlatL2(D)
            if self.res is not None:
                from faiss import index_cpu_to_gpu
                gpu_index = index_cpu_to_gpu(self.res, 0, gpu_index)

            nn_xinduce_idx = torch.empty(M - self.k, self.k, dtype=torch.int64)

            x = inducing_points.data.float().cpu().numpy()
            gpu_index.add(x[:self.k])
            for i in range(self.k, M):
                row = x[i][None, :]
                nn_xinduce_idx[i-self.k].copy_(
                    torch.from_numpy(gpu_index.search(row, self.k)[1][..., 0,:]).long().to(inducing_points.device)
                )
                gpu_index.add(row)
        return nn_xinduce_idx

    def to(self, device=None, dtype=None):
        if "cpu" in str(device):
            return self.cpu()
        elif "cuda" in str(device):
            return self.cuda()
        else:
            raise ValueError(f"Unknown device {device}")