from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths


mixed_ext = CppExtension(
    name='mixed',
    sources=['mixed.cpp'],
    include_dirs=include_paths(),
    language='cpp',
    extra_compile_args=['-I /usr/local/cuda/include']
)
setup(
    name='mixed_cpp',
    ext_modules=[mixed_ext],
    cmdclass={'build_ext': BuildExtension},
)
