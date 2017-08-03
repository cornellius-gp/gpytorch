import os
import torch
from torch.utils.ffi import create_extension

headers = ['gpytorch/csrc/fft.h']
sources = ['gpytorch/csrc/fft.c']
defines = []
with_cuda = False
libraries = ['fftw3']
library_dirs = []

if torch.cuda.is_available():
    cuda_home = os.getenv('CUDA_HOME') or '/usr/local/cuda'
    for base_dir in ['lib', 'lib64']:
        absolute_dir = os.path.join(cuda_home, base_dir)
        if os.path.exists(absolute_dir):
            library_dirs.append(absolute_dir)

    print('Including CUDA code.')
    headers += ['gpytorch/csrc/fft_cuda.h']
    sources += ['gpytorch/csrc/fft_cuda.c']
    defines += [('WITH_CUDA', None)]
    libraries += ['cufft']
    with_cuda = True

print(library_dirs)

ffi = create_extension(
    'gpytorch.libfft',
    headers=headers,
    sources=sources,
    define_macros=defines,
    libraries=libraries,
    library_dirs=library_dirs,
    with_cuda=with_cuda,
    package=True,
    relative_to=__file__,
)
