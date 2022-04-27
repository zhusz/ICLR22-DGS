# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

