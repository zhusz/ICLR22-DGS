# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def mode(a):
    u, c = np.unique(a, return_counts=True)
    return u[c.argmax()]


def checkModes():
    a = np.array([3, 2, 6, 1, 0, 7, 2, 7, 4, 4, 3, 3, 2], dtype=np.int32)
    b = np.stack([a, a + 1, a - 2], 0)
    print(mode(a))
    print(np.apply_along_axis(mode, 0, b))
    print(np.apply_along_axis(mode, 1, b))


if __name__ == '__main__':
    checkModes()
