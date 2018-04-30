from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def fft1(input):
    complex_input = torch.stack((input, input.new(input.shape).fill_(0)), dim=-1)
    return complex_input.fft(1)


def ifft1(input):
    complex_output = input.ifft(1)
    real_ind = complex_output.new(1).fill_(0).long()
    return complex_output.index_select(-1, real_ind).squeeze(-1)
