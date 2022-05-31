from .config_Shybrid import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict['hmRenderOm']['batchSize'] = 2
    config.datasetConfDict['omnidataBerkeley']['batchSize'] = 2

    config.datasetConfDict['hmRenderOm'] = {
        'dataset': 'hmRenderOm', 'trainingSplit': 'train', 'batchSize': 2,
        'class': 'HmRenderOmDataset',

        # Sampling
        'samplingMethodList': ['fullsurface', 'nearsurface', 'selfnonsurface',
                               'unarysurface', 'localEdgePairwiseSurface', 'localPlanarPairwiseSurface'],

        'numSamplingFinalFullsurface': 2048 * 16,

        'fullsurfaceNumNearestNeighbour': 5,
        'numSamplingFinalUnarysurface': 2048,
        'numSamplingSufficientUnarysurface': 2048 * 8,
        'numSamplingFinalLocalEdgePairwiseSurface': 512,  # pairs
        'numSamplingSufficientLocalEdgePairwiseSurface': 512 * 8,  # pairs
        'numSamplingFinalLocalPlanarPairwiseSurface': 512,  # pairs
        'numSamplingSufficientLocalPlanarPairwiseSurface': 512 * 8,  # pairs

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

    # wl
    wl = {}

    wl['lossOmniSsi'] = 1. * 0.2
    wl['lossOmniReg'] = 0.1 * 0.2
    wl['lossOmniVnl'] = 10. * 0.2

    wl['lossNearsurfaceClf'] = 1.
    wl['lossSelfnonsurfaceGradOccfloat'] = 0.01
    wl['lossDepthRegL1'] = 1.

    wl['lossUnarysurfaceNormal'] = 1.
    wl['lossLocalEdgePairwiseSurfaceNormal'] = 1.
    wl['lossLocalPlanarPairwiseSurfaceNormal'] = 1.

    config.wl = wl

    return config
