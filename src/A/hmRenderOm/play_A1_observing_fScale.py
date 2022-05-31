# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
from UDLv3 import udl
from BpredwoodA.testDataEntry.testDataEntryPool import getTestDataEntryDict
import numpy as np


def main():
    dataset = os.path.realpath(__file__).split('/')[-2]

    indChosen_val = getTestDataEntryDict(wishedTestDataNickName=['hmOfficialValSplit10'])['hmOfficialValSplit10']['indChosen']

    flagSplit = udl('_A01_flagSplit', dataset)
    m = udl('pkl_A1_m', dataset)

    A1 = udl('pkl_A1_', dataset)
    fScaleList = A1['fScaleList']

    fScaleList_val = fScaleList[indChosen_val]

    import ipdb
    ipdb.set_trace()
    print(1 + 1)

        
if __name__ == '__main__':
    main()
