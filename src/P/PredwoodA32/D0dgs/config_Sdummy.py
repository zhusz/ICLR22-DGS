from .config_Shybrid import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict['hmRenderOm']['ifDummy'] = True
    config.datasetConfDict['omnidataBerkeley']['ifDummy'] = True

    return config
