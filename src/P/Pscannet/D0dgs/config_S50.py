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

    # debug
    config.debug = False

    # datasetMetaConf  (can use a different one during demo time)
    config.datasetMetaConf = {
        # Sampling
        'samplingMethodList': ['fullsurface', 'nearsurface', 'selfnonsurface'],

        'numSamplingFinalFullsurface': 2048 * 16,

        'numSamplingFinalNearsurface': 2048,
        'numSamplingSufficientNearsurface': 2048 * 4,
        'nearsurfaceDeltaRange': 0.016 * 4,

        'numSamplingFinalSelfnonsurface': 512,
        'numSamplingSufficientSelfnonsurface': 512 * 4,
        'selfnonsurfaceDeltaRange': 0.008 * 4,

        'winWidth': 256,
        'winHeight': 192,
        'focalLengthWidth': 230.4,
        'focalLengthHeight': 230.4,

        'zNear': 1.e-4,

    }
    config.datasetMetaConf['fScaleWidth'] = \
        2. * config.datasetMetaConf['focalLengthWidth'] / config.datasetMetaConf['winWidth']
    config.datasetMetaConf['fScaleHeight'] = \
        2. * config.datasetMetaConf['focalLengthHeight'] / config.datasetMetaConf['winHeight']

    # datasetConfList  (can use a different one during demo time)
    datasetConf = copy.deepcopy(config.datasetMetaConf)
    datasetConf['did'] = 0
    datasetConf['dataset'] = 'scannetGivenRender'
    datasetConf['trainingSplit'] = 'train'
    datasetConf['batchSize'] = 32
    config.datasetConfList = [datasetConf]

    # Update datasetMetaConf - batchSizeTot  (can use a different one during demo time)
    config.datasetMetaConf['batchSizeTot'] = \
        sum([datasetConf['batchSize'] for datasetConf in config.datasetConfList])

    # orthogonal volume space
    config.r = float('inf')

    # encoder
    config.depthEncoderTag = 'resnet50_stride32'
    config.c_dim_specs = {'s32': 256, 's16': 256, 's8': 256, 's4': 256, 's2': 256}
    config.ifFinetunedFromAdelai = 1

    # optimizer
    config.lr = 1.e-4
    config.adam_epsilon = 1.e-8

    # wl
    wl = {}
    wl['lossNearsurfaceClf'] = 1.
    wl['lossSelfnonsurfaceGradOccfloat'] = 0.01
    wl['lossDepthRegL1'] = 1.
    config.wl = wl

    return config
