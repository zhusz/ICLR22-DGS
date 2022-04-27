# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .config_S50 import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)
    config.depthEncoderTag = 'resnext101_stride32x8d'
    config.c_dim_specs = {'s32': 256, 's16': 256, 's8': 256, 's4': 256, 's2': 256}  # not changed actually
    config.ifFinetunedFromAdelai = 1
    return config