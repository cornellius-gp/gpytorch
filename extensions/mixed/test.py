import torch
import mixed
import time
import numpy as np


# tensor cores need multiples of 8
m = 8**2
k = 8**5
n = 8**2
b = 8 

print("Testing accuracy...")
# nonbatched small
A = torch.randn((m, k), dtype=torch.float32, device="cuda")
B = torch.randn((k, n), dtype=torch.float32, device="cuda")
C = mixed.mm(A, B, None)
torch_C = A.half().mm(B.half())
true_C = A.mm(B)
print(f"mixed.mm residual: {(true_C - C).norm()}")
print(f"torch.mm residual: {(true_C - torch_C.float()).norm()}")


# batched small
A = torch.randn((b, m, k), dtype=torch.float32, device="cuda")
B = torch.randn((b, k, n), dtype=torch.float32, device="cuda")
C = mixed.bmm(A, B, None)
torch_C = A.half().bmm(B.half())
true_C = A.bmm(B)
print(f"mixed.bmm residual: {(true_C - C).norm()}")
print(f"torch.bmm residual: {(true_C - torch_C.float()).norm()}")


# timing
num_trials = 10
# non-batched
A = torch.randn(m, k, dtype=torch.float16, device="cuda")
B = torch.randn(k, n, dtype=torch.float16, device="cuda")

print(f"A has size {A.size()}")
print(f"B has size {B.size()}")

times = []
for i in range(num_trials):
    start = time.time()
    res = mixed.mm(A, B, None)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Non-inplace mixed.mm time ({num_trials} trials): {times.mean():.2e}")

times = []
for i in range(num_trials):
    start = time.time()
    res = torch.mm(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Non-inplace torch.mm time ({num_trials} trials): {times.mean():.2e}")

A = A.float()
B = B.float()
times = []
for i in range(num_trials):
    start = time.time()
    res = torch.mm(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Non-inplace torch.mm time fp32 ({num_trials} trials): {times.mean():.2e}")
A = A.half()
B = B.half()

times = []
for i in range(num_trials):
    C_ = torch.empty(m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = mixed.mm(A, B, C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Inplace mixed.mm time ({num_trials} trials): {times.mean():.2e}")

times = []
for i in range(num_trials):
    C_ = torch.empty(m, n, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = torch.mm(A, B, out=C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Inplace torch.mm time ({num_trials} trials): {times.mean():.2e}")

A = A.float()
B = B.float()
times = []
for i in range(num_trials):
    C_ = torch.empty(m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = torch.mm(A, B, out=C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Inplace torch.mm time fp32 ({num_trials} trials): {times.mean():.2e}")
A = A.half()
B = B.half()

del A; del B; del C; del C_; del true_C;

# batched
A = torch.randn(b, m, k, dtype=torch.float16, device="cuda")
B = torch.randn(b, k, n, dtype=torch.float16, device="cuda")

print(f"A has size {A.size()}")
print(f"B has size {B.size()}")

times = []
for i in range(num_trials):
    start = time.time()
    res = mixed.bmm(A, B, None)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace mixed.bmm time ({num_trials} trials): {times.mean():.2e}")

times = []
for i in range(num_trials):
    start = time.time()
    res = torch.bmm(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace torch.bmm time ({num_trials} trials): {times.mean():.2e}")

times = []
for i in range(num_trials):
    start = time.time()
    res = torch.matmul(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace torch.matmul time ({num_trials} trials): {times.mean():.2e}")

A = A.float()
B = B.float()
times = []
for i in range(num_trials):
    start = time.time()
    res = torch.bmm(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace torch.bmm time fp32 ({num_trials} trials): {times.mean():.2e}")
A = A.half()
B = B.half()

A = A.float()
B = B.float()
times = []
for i in range(num_trials):
    start = time.time()
    res = torch.matmul(A, B)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace torch.matmul time fp32 ({num_trials} trials): {times.mean():.2e}")
A = A.half()
B = B.half()

times = []
for i in range(num_trials):
    C_ = torch.empty(b, m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = mixed.bmm(A, B, C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched inplace mixed.bmm time ({num_trials} trials): {times.mean():.2e}")

times = []
for i in range(num_trials):
    C_ = torch.empty(b, m, n, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = torch.bmm(A, B, out=C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched inplace torch.bmm time ({num_trials} trials): {times.mean():.2e}")

A = A.float()
B = B.float()
times = []
for i in range(num_trials):
    C_ = torch.empty(b, m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = torch.bmm(A, B, out=C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched inplace torch.bmm time fp32 ({num_trials} trials): {times.mean():.2e}")
A = A.half()
B = B.half()
