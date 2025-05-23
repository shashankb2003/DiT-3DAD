import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['--compiler-bindir=/usr/bin/gcc'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
