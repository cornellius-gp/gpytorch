from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .. import libfft


def fft1(input):
    # [..., d]
    orig_size = input.size()
    orig_type = input.type()

    input = input.view(-1, input.size(-1))
    n, d = input.size()

    output = input.new().resize_(n, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft1_r2c_cuda(input, output)
    else:
        output = output.float()
        libfft.fft1_r2c(input.float(), output)

    if len(orig_size) > 1:
        output_size = list(orig_size[:-1]) + [(d // 2) + 1, 2]
    else:
        output_size = [(d // 2) + 1, 2]

    return output.view(*output_size).type(orig_type)


def ifft1(input, size=None):
    # [..., d, 2]
    orig_type = input.type()

    if not size:
        size = list(input.size())[:-1]
        d = (size[-1] - 1) * 2
        size[-1] = d
    else:
        d = size[-1]
    input = input.view(-1, *input.size()[-2:])

    output = input.new().resize_(input.size(0), d)
    if input.is_cuda:
        libfft.fft1_c2r_cuda(input, output)
    else:
        output = output.float()
        libfft.fft1_c2r(input.float(), output)
    output.div_(d)
    return output.view(size).type(orig_type)
