import torch
import mixed
import time
import numpy as np

m = 3
k = 4
n = 5
b = 2

print("Testing small matrices...")
# nonbatched small
A = torch.arange(0, m * k, dtype=torch.float16, device="cuda").view(m, k)
B = torch.arange(0, n * k, dtype=torch.float16, device="cuda").view(k, n)
C_ = torch.empty(m, n, dtype=torch.float32, device="cuda")
C = mixed.mm(A, B, None)
C_ = mixed.mm(A, B, C_)
true_C = A.mm(B)
if torch.allclose(true_C, C.half()) and torch.allclose(true_C, C_.half()):
    print("mixed.mm passed!")

# batched small
A = torch.arange(0, b * m * k, dtype=torch.float16, device="cuda").view(b, m, k)
B = torch.arange(0, b * n * k, dtype=torch.float16, device="cuda").view(b, k, n)
C_ = torch.empty(b, m, n, dtype=torch.float32, device="cuda")
C = mixed.bmm(A, B, None)
C_ = mixed.bmm(A, B, C_)
true_C = A.bmm(B)
if torch.allclose(true_C, C.half()) and torch.allclose(true_C, C_.half()):
    print("mixed.bmm passed!")

# nonbatched small fp32
A = torch.arange(0, m * k, dtype=torch.float32, device="cuda").view(m, k)
B = torch.arange(0, n * k, dtype=torch.float32, device="cuda").view(k, n)
C_ = torch.empty(m, n, dtype=torch.float32, device="cuda")
C = mixed.mm(A, B, None)
C_ = mixed.mm(A, B, C_)
true_C = A.mm(B)
if torch.allclose(true_C, C) and torch.allclose(true_C, C_):
    print("mixed.mm with casting passed!")

# batched small fp32
A = torch.arange(0, b * m * k, dtype=torch.float32, device="cuda").view(b, m, k)
B = torch.arange(0, b * n * k, dtype=torch.float32, device="cuda").view(b, k, n)
C_ = torch.empty(b, m, n, dtype=torch.float32, device="cuda")
C = mixed.bmm(A, B, None)
C_ = mixed.bmm(A, B, C_)
true_C = A.bmm(B)
if torch.allclose(true_C, C) and torch.allclose(true_C, C_):
    print("mixed.bmm with casting passed!")



m = 2**15
k = 2**9
n = 2**12

print("Testing large matrices...")
num_trials = 10
# non-batched
A = torch.randn(m, k, dtype=torch.float16, device="cuda")
B = torch.randn(k, n, dtype=torch.float16, device="cuda")

print(f"A has size {A.size()}")
print(f"B has size {A.size()}")

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

del A; del B; del C; del C_; del true_C;

b = 2**3
# batched
A = torch.randn(b, m, k, dtype=torch.float16, device="cuda")
B = torch.randn(b, k, n, dtype=torch.float16, device="cuda")

print(f"A has size {A.size()}")
print(f"B has size {A.size()}")

times = []
for i in range(num_trials):
    start = time.time()
    res = mixed.bmm(A, B, None)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched non-inplace mixed.mm time ({num_trials} trials): {times.mean():.2e}")

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
    C_ = torch.empty(b, m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    res = mixed.bmm(A, B, C_)
    torch.cuda.synchronize()
    times.append(time.time() - start)
times = np.array(times)
print(f"Batched inplace mixed.mm time ({num_trials} trials): {times.mean():.2e}")

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
