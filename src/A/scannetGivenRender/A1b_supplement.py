# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    A1 = udl('pkl_A1_', dataset)
    rawPoseMat = A1['rawPoseMat']

    # rawPoseMat = rawPoseMat[:100, :, :]

    E = rawPoseMat[:, :3, 3]
    L = E + np.dot(
        rawPoseMat[:, :3, :3],
        np.array([0, 0, 1], dtype=np.float32)
    )
    U = np.dot(
        rawPoseMat[:, :3, :3],
        np.array([0, -1, 0], dtype=np.float32)
    )

    cam = ELU2cam(np.concatenate([E, L, U], 1))
    cam2world = np.linalg.inv(cam)

    delta = rawPoseMat - cam2world
    assert np.nanmean(np.abs(delta)) < 1.e-5

    pSaveMat(projRoot + 'v/A/%s/A1b_supplement.mat' % dataset, {
        'EWorld': E, 'LWorld': L, 'UWorld': U,
    })


if __name__ == '__main__':
    main()
