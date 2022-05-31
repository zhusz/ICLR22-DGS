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
    m = A1['m']

    flagSplit = np.zeros((m, ), dtype=np.int32)
    flagSplit[:800] = 1
    flagSplit[800:900] = 2
    assert np.all(flagSplit > 0)

    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
    })


if __name__ == '__main__':
    main()


'''
def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    dataset_house = A1['houseDataset']
    assert dataset_house == 'hm'
    A1_house = udl('pkl_A1_', dataset_house)
    

    flag123 = np.zeros((m, 3), dtype=bool)
    flagSplit = np.zeros((m, ), dtype=np.int32)

    for j in range(m):
        if j % 100 == 0:
            print('Processing A01_ind for %s: %d / %d')

        houseID0 = int(houseIDList[j])
        pointID0 = int(pointIDList[j])
        viewID0 = int(viewIDList[j])
'''
