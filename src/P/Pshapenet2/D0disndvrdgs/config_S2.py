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
        'dataset': 'corenetChoySingleRendering', 'trainingSplit': 'trainval', 'batchSize': 32,
        'class': 'CorenetChoySingleRenderingDataset',

        # sampling
        'samplingMethodList': ['nearsurfacemass', 'nearsurfaceair', 'nonsurface'],

        'numSamplingFinalNearsurfacemass': 1024,
        'nearsurfacemassRForNeighbouringRange': 2,  # roughly 0.015 for deltaRange (2 * 1/128)
        'nearsurfacemassGridPurtFactor': 1.,

        'numSamplingFinalNearsurfaceair': 1024,
        'nearsurfaceairRForNeighbouringRange': 2,  # roughly 0.015 for deltaRange (2 * 1/128)
        'nearsurfaceairGridPurtFactor': 1.,

        'numSamplingFinalNonsurface': 512,
        'nonsurfaceRForNeighbouringRange': 1,  # roughly 0.007 for delta range (1 * 1/128)
        'nonsurfaceGridPurtFactor': 1.,

        'winHeight': 256,
        'winWidth': 256,
    }
    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf['dataset']

    # encoder
    config.encoderTag = 'resnet50'

    # lr
    config.lr = 4.e-4
    config.adam_epsilon = 1.e-4

    # wl
    wl = {}
    wl['lossNearsurfacemassClf'] = 1.
    wl['lossNearsurfaceairClf'] = 1.
    wl['lossNonsurfaceClfGradZero'] = 0.01
    config.wl = wl

    return config
