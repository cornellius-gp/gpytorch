from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths


mixed_ext = CppExtension(
    name='mixed_cpp',
    sources=['mixed.cpp'],
    include_dirs=include_paths() + ['/usr/local/cuda/include'],
    language='cpp',
)
setup(
    name='mixed_cpp',
    ext_modules=[mixed_ext],
    cmdclass={'build_ext': BuildExtension},
    py_modules=['mixed'],
)
