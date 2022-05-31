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
    config.datasetConfDict['hmRenderOm'] = {
        'dataset': 'hmRenderOm', 'trainingSplit': 'train', 'batchSize': 32,
        'class': 'HmRenderOmDataset',

        # Sampling
        'samplingMethodList': ['fullsurface', 'nearsurface', 'selfnonsurface'],

        'numSamplingFinalFullsurface': 2048 * 16,

        'numSamplingFinalNearsurface': 2048,
        'numSamplingSufficientNearsurface': 2048 * 8,
        'nearsurfaceDeltaRange': 0.016 * 4,

        'numSamplingFinalSelfnonsurface': 512,
        'numSamplingSufficientSelfnonsurface': 512 * 8,
        'selfnonsurfaceDeltaRange': 0.008 * 4,

        'winWidth': 256,
        'winHeight': 256,
        # These params are no longer from only the datasetConf (because they are now likely to be variant on different samples)
        # although in this S, they are still fixed.
        # But you can no longer assume this anymore.
        # 'focalLengthWidth': 230.4, 
        # 'focalLengthHeight': 230.4,
        # 'fScaleWidth': 1.8,
        # 'fScaleHeight': 1.8,

        'zNear': 1.e-6,

        'ifNeedPixelAugmentation': 0,
        'ifNeedCroppingAugmentation': 1,
        'fScaleWidthDesiredFixed': 1.8,
        'fScaleHeightDesiredFixed': 1.8,
        'ifNeedMirrorAugmentation': 0,

        'ifDummy': False,
    }
    config.datasetConfDict['omnidataBerkeley'] = {
        'dataset': 'omnidataBerkeley', 'trainingSplit': 'train', 'batchSize': 8,
        'class': 'OmnidataBerkeleyDataset',

        'componentFrequency': {
            'taskonomy': 0.5, 'replica': 0.1, 'gso': 0.05,
            'hypersim': 0.3, 'blendedMVS': 0.05,
        },

        'winWidth': 256, 'winHeight': 256,

        'ifDummy': False,
    }
    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf['dataset']

    # encoder
    config.omnitoolsBackbone = 'vitb_rn50_384'
    config.c_dim_specs = {'s1': 256, 's2': 256, 's3': 256, 's4': 256}
    config.ifFinetunedFromOmnitools = 1

    # optimizer
    config.lr = 1.e-4
    config.adam_epsilon = 1.e-8

    # wl
    wl = {}

    wl['lossOmniSsi'] = 1. * 0.2
    wl['lossOmniReg'] = 0.1 * 0.2
    wl['lossOmniVnl'] = 10. * 0.2

    wl['lossNearsurfaceClf'] = 1.
    wl['lossSelfnonsurfaceGradOccfloat'] = 0.01
    wl['lossDepthRegL1'] = 1.
    config.wl = wl

    return config
