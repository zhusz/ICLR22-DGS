from .config_Shybrid import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict['scannetGivenRender']['batchSize'] = 2
    config.datasetConfDict['omnidataBerkeley']['batchSize'] = 2

    return config
