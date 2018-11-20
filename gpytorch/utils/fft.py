#!/usr/bin/env python3

import torch


def fft1(input):
    complex_input = torch.stack((input, torch.zeros_like(input)), dim=-1)
    return complex_input.fft(1)


def ifft1(input):
    complex_output = input.ifft(1)
    real_ind = torch.tensor(0, dtype=torch.long, device=input.device)
    return complex_output.index_select(-1, real_ind).squeeze(-1)
