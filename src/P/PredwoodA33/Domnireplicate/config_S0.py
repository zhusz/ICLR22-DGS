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
    config.datasetConfDict['omnidataBerkeley'] = {
        'dataset': 'omnidataBerkeley', 'trainingSplit': 'train', 'batchSize': 8,
        'class': 'OmnidataBerkeleyDataset',

        'componentFrequency': {
            'taskonomy': 0.2, 'replica': 0.2, 'gso': 0.2,
            'hypersim': 0.2, 'blendedMVS': 0.2,
        },

        'winWidth': 384, 'winHeight': 384,
    }
    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf['dataset']

    # encoder
    config.omnitoolsBackbone = 'vitb_rn50_384'
    config.ifFinetunedFromOmnitools = 0  # replicate the training of omnidata_tools

    # optimizer
    config.lr = 1.e-5  # set according to omnidata_tools
    config.adam_epsilon = 1.e-8

    # wl
    wl = {}
    wl['lossOmniSsi'] = 1.
    wl['lossOmniReg'] = 0.1
    wl['lossOmniVnl'] = 10.
    config.wl = wl

    return config
