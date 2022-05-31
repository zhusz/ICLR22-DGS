from .config_Shybrid import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict['scannetGivenRender']['batchSize'] = 16

    return config
