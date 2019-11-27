import torch
import gpytorch
from gpytorch.kernels.kernel import Distance
import time

dist = Distance()
n_trials = 10
total_time = 0
for i in range(n_trials):
    x = torch.randn(8**4, 8, dtype=torch.float32, device="cuda:0")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with gpytorch.settings.use_fp16_dist():
        res = dist._sq_dist(x, x, False, True)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    total_time += elapsed_time_ms

print(f"Average time(ms): {total_time / n_trials}")
