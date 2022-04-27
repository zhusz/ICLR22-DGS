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
    config.R = R  # used by datasetConf['sampleID']

    # datasetMetaConf  (can use a different one during demo time)
    config.datasetMetaConf = {}
    config.datasetMetaConf.update({
        # sampling
        # 'numSamplingFinalGridpoint': 32 ** 3,  # 128 * 128 * 128,
        'gridPointLength': 128,

        # winSize
        'winHeight': 256,
        'winWidth': 256,
    })
    config.datasetMetaConf['numSamplingFinalGridpoint'] = \
        config.datasetMetaConf['gridPointLength'] ** 3

    # datasetConfList  (can use a different one during demo time)
    datasetConf = copy.deepcopy(config.datasetMetaConf)
    datasetConf['did'] = 0
    datasetConf['dataset'] = 'corenetChoySingleRendering'
    datasetConf['batchSize'] = 4
    datasetConf['trainingSplit'] = 'trainval'
    config.datasetConfList = [datasetConf]

    # Update datasetMetaConf - batchSizeTot  (can use a different one during demo time)
    config.datasetMetaConf['batchSizeTot'] = \
        sum([datasetConf['batchSize'] for datasetConf in config.datasetConfList])

    # lr
    config.lr = 4.e-4
    config.adam_epsilon = 1.e-4

    # wl
    wl = {}
    wl['lossGridpointIou'] = 1.
    config.wl = wl

    return config
