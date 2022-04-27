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
sys.path.append(projRoot + 'src/P/')
from .testDataEntry.testDataEntryPool import getTestDataEntryDict
from .approachEntryPool import getApproachEntryDict
import torch
import random
import numpy as np


def main():
    # set your set - testDataNickName
    testDataNickName = os.environ.get('DS', 'corenetSingleOfficialTestSplitFirstOnePerCent')
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    approachEntryDict = getApproachEntryDict()
    testDataEntry = testDataEntryDict[testDataNickName]

    # random seed and precision
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # so your result should be first put into np.float16 and then put back to np.float32
    # to make sure every time you run your model for getting the same result
    # for benchmarking

    # specify
    approachEntry = approachEntryDict[os.environ['MC']]
    if approachEntry['scriptTag'] == '1D':
        from .csvShapenetEntry.Bpshapenet2_1D_corenet import testApproachOnTestData
    elif approachEntry['scriptTag'] == '1D128':
        from .csvShapenetEntry.Bpshapenet2_1D128_corenet import testApproachOnTestData
    else:
        raise NotImplementedError('Unknown approach script tag: %s' %
                                  approachEntry['scriptTag'])
    testApproachOnTestData(testDataEntry, approachEntry)


if __name__ == '__main__':
    main()

