# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
import os
import sys
from configs_registration import getConfigGlobal
from datasets_registration import datasetDict
import numpy as np


from .corenetChoySingleRenderingDataset import CorenetChoySingleRenderingDataset


bt = lambda s: s[0].upper() + s[1:]


def getTestDataEntryDict(**kwargs):
    wishedTestDataNickName = kwargs['wishedTestDataNickName']
    assert type(wishedTestDataNickName) is list or wishedTestDataNickName is None

    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    sys.path.append(projRoot + 'src/P/')
    testDataEntryDict = {}

    # --------------------- corenetSingleOfficialTestSplitFirstOnePerCent --------------------- #
    if (wishedTestDataNickName is None) or \
            'corenetSingleOfficialTestSplitFirstOnePerCent' in wishedTestDataNickName:
        meta = {}
        datasetConf = {}
        datasetConf['winHeight'] = 256
        datasetConf['winWidth'] = 256
        datasetConf['dataset'] = 'corenetChoySingleRendering'
        datasetObj = CorenetChoySingleRenderingDataset(
            datasetConf, projRoot=projRoot,
            datasetSplit='test',
            cudaDevice='cuda:0',
        )
        flagSplit = datasetObj.flagSplit
        indChosen = np.where(flagSplit == 3)[0][:1898]
        indVisChosen = indChosen[[0, 100, 200, 300]]
        indChosen = indChosen.tolist()
        indVisChosen = indVisChosen.tolist()
        entry = dict(
            testDataNickName='corenetSingleOfficialTestSplitFirstOnePerCent',
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indChosen, indVisChosen, entry

    # --------------------- corenetSingleOfficialTestSplit --------------------- #
    if (wishedTestDataNickName is None) or \
            'corenetSingleOfficialTestSplit' in wishedTestDataNickName:
        meta = {}
        datasetConf = {}
        datasetConf['winHeight'] = 256
        datasetConf['winWidth'] = 256
        datasetConf['dataset'] = 'corenetChoySingleRendering'
        datasetObj = CorenetChoySingleRenderingDataset(
            datasetConf, projRoot=projRoot,
            datasetSplit='test',
            cudaDevice='cuda:0',
        )
        flagSplit = datasetObj.flagSplit
        indChosen = np.where(flagSplit == 3)[0]
        indVisChosen = indChosen[[0, 100, 200, 300]]
        indChosen = indChosen.tolist()
        indVisChosen = indVisChosen.tolist()
        entry = dict(
            testDataNickName='corenetSingleOfficialTestSplit',
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indChosen, indVisChosen, entry

    # ------------------- More to Come -------------------- #

    return testDataEntryDict
