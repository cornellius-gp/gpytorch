import torch
import mixed

m = 4
k = 2
n = 3

A = torch.arange(0, m * k, dtype=torch.float16, device="cuda").view(m, k)
B = torch.arange(0, n * k, dtype=torch.float16, device="cuda").view(k, n)
import pdb; pdb.set_trace()
C = mixed.matmul(A, B)
