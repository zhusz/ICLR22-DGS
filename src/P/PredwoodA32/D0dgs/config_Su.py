from .config_Shybrid import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    ''' Config pool
    config.datasetConfDict['hmRenderOm']['samplingMethodList'] = [
        'fullsurface', 'nearsurface', 'selfnonsurface',
        'unarysurface', 'localEdgePairwiseSurface', 'localPlanarPairwiseSurface']
    config.datasetConfDict['hmRenderOm']['fullsurfaceNumNearestNeighbour'] = 5
    config.datasetConfDict['hmRenderOm']['numSamplingFinalUnarysurface'] = 2048
    config.datasetConfDict['hmRenderOm']['numSamplingSufficientUnarysurface'] = 2048 * 8
    config.datasetConfDict['hmRenderOm']['numSamplingFinalLocalEdgePairwiseSurface'] = 512
    config.datasetConfDict['hmRenderOm']['numSamplingSufficientLocalEdgePairwiseSurface'] = 512 * 8
    config.datasetConfDict['hmRenderOm']['numSamplingFinalLocalPlanarPairwiseSurface'] = 512
    config.datasetConfDict['hmRenderOm']['numSamplingSufficientLocalPlanarPairwiseSurface'] = 512 * 8

    config.wl['lossUnarysurfaceNormal'] = 1.
    config.wl['lossLocalEdgePairwiseSurfaceNormal'] = 1.
    config.wl['lossLocalPlanarPairwiseSurfaceNormal'] = 1.
    '''

    config.datasetConfDict['hmRenderOm']['samplingMethodList'] = [
        'fullsurface', 'nearsurface', 'selfnonsurface',
        'unarysurface']

    config.datasetConfDict['hmRenderOm']['numSamplingFinalUnarysurface'] = 1024
    config.datasetConfDict['hmRenderOm']['numSamplingSufficientUnarysurface'] = 1024 * 8

    config.wl['lossUnarysurfaceNormal'] = 0.5

    return config

