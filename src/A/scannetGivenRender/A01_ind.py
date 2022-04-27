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
import numpy as np
import matplotlib.pyplot as plt
from codes_py.np_ext.np_image_io_v1 import imread
import cv2
from codes_py.np_ext.mat_io_v1 import pSaveMat


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    A1 = udl('pkl_A1_', dataset)
    dataset_house = A1['houseDataset']
    rawPoseMat = A1['rawPoseMat']
    A1b = udl('_A1b_', dataset)
    E = A1b['EWorld']
    L = A1b['LWorld']
    U = A1b['UWorld']
    A01_house = udl('_A01_', dataset_house)
    flagSplit_house = A01_house['flagSplit']

    # weird camera settings (now will not deleting any case from here)
    info = np.concatenate([
        A1[k] for k in ['fxywxyColor', 'ppxyColor', 'fxywxyDepth', 'ppxyDepth']
    ], 1)
    houseIDList = A1['houseIDList']

    # inf in camera pose
    ELU = np.concatenate([E, L, U], 1)
    maskELU = np.isfinite(ELU)
    maskRaw = np.isfinite(rawPoseMat).reshape((-1, 16))
    fELU = np.any(maskELU, axis=1)
    fRaw = np.any(maskRaw, axis=1)
    assert np.all(fELU == fRaw)

    # filter out very small intrinsics / winSize
    wrong0 = info[:, 0] < 1000
    wrong1 = info[:, 1] < 1000
    wrong2 = info[:, 2] < 1000
    wrong3 = info[:, 3] < 500
    wrong4 = info[:, 4] < 500
    wrong5 = info[:, 5] < 250
    wrongs = [wrong0, wrong1, wrong2, wrong3, wrong4, wrong5]
    for k in range(5):
        assert np.all(wrongs[k] == wrongs[k + 1])

    flagSplit = flagSplit_house[houseIDList]
    flagSplit[fRaw == 0] = 0
    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
    })


if __name__ == '__main__':
    main()
