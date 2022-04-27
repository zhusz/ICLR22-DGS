# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run in parallel:
# for g in {0..9}; do MC=P18dgsD0S101RKI9WOn1E J1=$((100*g)) J2=$((100*(g+1))) CUDA_VISIBLE_DEVICES=$g PYOPENGL_PLATFORM=osmesa python -m Bp18singleView.run & done

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from .testDataEntry.testDataEntryPool import getTestDataEntryDict
from .approachEntryPool import getApproachEntryDict
import torch
import random
import numpy as np


def main():
    testDataNickName = os.environ['DS']
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    testDataEntry = testDataEntryDict[testDataNickName]
    approachEntryDict = getApproachEntryDict()

    # random seed and precision
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # specify
    approachEntry = approachEntryDict[os.environ['MC']]
    if approachEntry['scriptTag'] == '1E':
        from .csvGeometry3DEntry.Bpscannet_1E_noPostProcess import testApproachOnTestData
    elif approachEntry['scriptTag'] == '1Epp1':
        from .csvGeometry3DEntry.Bpscannet_1Epp1_viaPP1 import testApproachOnTestData
    elif approachEntry['scriptTag'] == '1G':
        from .csvGeometry3DEntry.Bpscannet_1G_ext import testApproachOnTestData
    else:
        raise NotImplementedError('Unknown approach script tag: %s' %
                                  approachEntry['scriptTag'])
    testApproachOnTestData(testDataEntry, approachEntry)


if __name__ == '__main__':
    main()

