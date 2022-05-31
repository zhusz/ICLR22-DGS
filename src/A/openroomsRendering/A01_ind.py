# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    A1 = udl('pkl_A1_', dataset)
    dataset_house = A1['houseDataset']
    assert dataset_house == 'openrooms', dataset_house
    m = A1['m']
    houseIDList = A1['houseIDList']

    flagSplit_house = udl('mat_A01_flagSplit', dataset_house)

    flagSplit = flagSplit_house[houseIDList]

    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
    })


if __name__ == '__main__':
    main()

