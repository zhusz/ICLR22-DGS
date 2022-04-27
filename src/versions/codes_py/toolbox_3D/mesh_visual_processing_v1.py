# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
from matplotlib.cm import get_cmap
import numpy as np


# meta data for colorPool_nyu40structureDark_airWhite
nyu40RetrieveList = udl('_Z_semanticNYU')['nyu40RetrieveList']
nyu40Dict = udl('_Z_semanticNYU')['nyu40Dict']
cmap = get_cmap('spring')
colorPool_nyu40structureDark_airWhite = np.array([np.array(cmap(float(j) / len(nyu40Dict))[:3], dtype=np.float32) if nyu40RetrieveList[j] not in ['wall', 'floor', 'window', 'mirror', 'floor mat', 'ceiling', 'otherstructure'] else np.array([0.1, 0.1, 0.1], dtype=np.float32) for j in range(len(nyu40Dict))], dtype=np.float32)
colorPool_nyu40structureDark_airWhite[0, :] = 1.  # the air class turns to white  # This is not useful for mesh vis (since mesh has no air), but useful for query ball vis.


def meshNyu40IDToColor_structureDark(faceNyu40ID0):
    assert 'int' in str(faceNyu40ID0.dtype)
    shape = faceNyu40ID0.shape

    faceNyu40IDColor0 = colorPool_nyu40structureDark_airWhite[faceNyu40ID0.reshape(-1), :]
    shape3 = list(shape) + [3]
    faceNyu40IDColor0 = faceNyu40IDColor0.reshape(shape3)
    return faceNyu40IDColor0
