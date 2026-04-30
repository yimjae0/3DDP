#python3 setup.py install
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Python 3.8 headers include crypt.h which lives in $CONDA_PREFIX/include
# when libxcrypt is installed via conda-forge. Add it explicitly.
_conda_inc = [os.path.join(os.environ['CONDA_PREFIX'], 'include')] \
    if 'CONDA_PREFIX' in os.environ else []

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',

            'src/ballquery/ballquery_cuda.cpp',
            'src/ballquery/ballquery_cuda_kernel.cu',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/knnquery_heap/knnquery_heap_cuda.cpp',
            'src/knnquery_heap/knnquery_heap_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/grouping_int/grouping_int_cuda.cpp',
            'src/grouping_int/grouping_int_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',

            'src/labelstat/labelstat_cuda.cpp',
            'src/labelstat/labelstat_cuda_kernel.cu',

            'src/featuredistribute/featuredistribute_cuda.cpp',
            'src/featuredistribute/featuredistribute_cuda_kernel.cu'
        ],
                      include_dirs=_conda_inc,
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
