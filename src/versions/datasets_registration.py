from easydict import EasyDict
from collections import OrderedDict


def getDatasetListDict():
    datasetList = [
                   'demo1',

                   'corenetChoySingleRendering',

                   'scannetGivenRender',

                   'freedemo1',

                   'mpv1GivenRenderHzDd',

                   'omnidataBerkeley',
                   ]
    datasetDict = {}
    for j in range(len(datasetList)):
        datasetDict[datasetList[j]] = j
    return datasetList, datasetDict


datasetList, datasetDict = getDatasetListDict()
# del datasetList

datasetRetrieveList = {}
for dataset in datasetDict.keys():
    i = int(datasetDict[dataset])
    datasetRetrieveList[i] = dataset

