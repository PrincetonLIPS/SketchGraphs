import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [
    CUDAExtension(
        'sketchgraphs_models.torch_extensions._native_component',
        sources=[os.path.join('cpp/torch', fn) for fn in [
            'module.cpp',
            'repeat_interleave.cpp',
            'repeat_interleave_cuda.cu',
            'segment_logsumexp_backward.cu',
            'segment_logsumexp_cuda.cu',
            'segment_logsumexp.cpp',
            'segment_pool_cuda.cu',
            'segment_pool.cpp'
        ]],
        include_dirs=[os.path.join(os.getcwd(), 'cpp/lib/cub-1.8.0/')],
        extra_compile_args={
            'cxx': ['-g', '-DAT_PARALLEL_OPENMP', '-fopenmp'],
            'nvcc': []
        })
]

setup(
    name='sketchgraphs',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension}
)