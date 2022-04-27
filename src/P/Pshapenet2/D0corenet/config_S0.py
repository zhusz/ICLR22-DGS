# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from easydict import EasyDict
from collections import OrderedDict
import numpy as np
import copy


def getConfigFunc(P, D, S, R, **kwargs):  # Abstract Config
    config = EasyDict()
    config.P = P
    config.D = D
    config.S = S
    config.R = R

    # debug
    config.debug = False

    config.datasetConfDict = OrderedDict([])
    config.datasetConfDict['corenetChoySingleRendering'] = {
        'dataset': 'corenetChoySingleRendering', 'trainingSplit': 'trainval', 'batchSize': 4,
        'class': 'CorenetChoySingleRenderingDataset',

        'gridPointLength': 128,
        'numSamplingFinalGridpoint': 128 ** 3,

        'winHeight': 256,
        'winWidth': 256,
    }
    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf['dataset']

    # lr
    config.lr = 4.e-4
    config.adam_epsilon = 1.e-4

    # wl
    wl = {}
    wl['lossGridpointIou'] = 1.
    config.wl = wl

    return config
