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

    for j in range(m):
        if j % 1000 == 0 or j < 10:
            print('Running check_R17_completeness for %s: %d / %d' % (dataset, j, m))

        if flagSplit[j] != 1 and j not in indChosen_val:
            continue

        assert os.path.isfile(projRoot + 'v/R/%s/R17/%08d.npz' % (dataset, j)), (j, flagSplit[j])


def play():
    dataset = os.path.realpath(__file__).split('/')[-2]

    indChosen_val = getTestDataEntryDict(wishedTestDataNickName=['hmOfficialValSplit10'])['hmOfficialValSplit10']['indChosen']

    flagSplit = udl('_A01_flagSplit', dataset)
    m = udl('pkl_A1_m', dataset)

    indChosen = np.where(flagSplit == 1)[0].tolist() + indChosen_val
    print(len(indChosen))


if __name__ == '__main__':
    play()
