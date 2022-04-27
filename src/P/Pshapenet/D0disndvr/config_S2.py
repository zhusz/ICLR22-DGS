# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from easydict import EasyDict
import numpy as np
import copy


def getConfigFunc(P, D, S, R, **kwargs):  # Abstract Config
    config = EasyDict()
    config.P = P
    config.D = D
    config.S = S
    config.R = R

    # datasetMetaConf  (can use a different one during demo time)
    config.datasetMetaConf = {}
    config.datasetMetaConf.update({
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

        # all need to be pertubed within a single grid

        # winSize
        'winHeight': 256,
        'winWidth': 256,
    })

    # datasetConfList  (can use a different one during demo time)
    datasetConf = copy.deepcopy(config.datasetMetaConf)
    datasetConf['did'] = 0
    datasetConf['dataset'] = 'corenetChoySingleRendering'
    datasetConf['batchSize'] = 32
    datasetConf['trainingSplit'] = 'trainval'
    config.datasetConfList = [datasetConf]

    # Update datasetMetaConf - batchSizeTot  (can use a different one during demo time)
    config.datasetMetaConf['batchSizeTot'] = \
        sum([datasetConf['batchSize'] for datasetConf in config.datasetConfList])

    # encoder
    config.encoderTag = 'resnet50'

    # lr
    config.lr = 4.e-4
    config.adam_epsilon = 1.e-4

    # wl
    wl = {}
    wl['lossNearsurfacemassClf'] = 1.
    wl['lossNearsurfaceairClf'] = 1.
    wl['lossNonsurfaceClf'] = 1.
    config.wl = wl

    return config


