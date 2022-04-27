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


from .scannetGivenRenderDataset import ScannetMeshCache, ScannetGivenRenderDataset
from .submissionTimeScannetGivenRenderDataset import \
    SubmissionTimeScannetMeshCache, SubmissionTimeScannetGivenRenderDataset
from .demoRenderDataset import DemoRenderDataset, FreeDemoRenderDataset


bt = lambda s: s[0].upper() + s[1:]


def getTestDataEntryDict(**kwargs):
    wishedTestDataNickName = kwargs['wishedTestDataNickName']
    assert type(wishedTestDataNickName) is list or wishedTestDataNickName is None

    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    sys.path.append(projRoot + 'src/P/')
    testDataEntryDict = {}

    # --------------------- scannetOfficialTestSplit ------------------ #
    if (wishedTestDataNickName is None) or \
            any([_.startswith('scannetOfficialTestSplit') for _ in wishedTestDataNickName]):
        meta = {}
        meta['scannetMeshCache'] = ScannetMeshCache()
        datasetConf = {}
        datasetConf['did'] = 0
        datasetConf['dataset'] = 'scannetGivenRender'
        datasetConf['winWidth'] = 256
        datasetConf['winHeight'] = 192
        datasetConf['focalLengthWidth'] = 230.4
        datasetConf['focalLengthHeight'] = 230.4
        datasetConf['fScaleWidth'] = 2. * datasetConf['focalLengthWidth'] / datasetConf['winWidth']
        datasetConf['fScaleHeight'] = 2. * datasetConf['focalLengthHeight'] / datasetConf['winHeight']
        datasetConf['zNear'] = 1.e-6
        datasetObj = ScannetGivenRenderDataset(
            datasetConf, projRoot=projRoot, datasetSplit='test',
            R='RlocalNono',  # read from /shared/group as usual
        )
        flagSplit = datasetObj.flagSplit
        indTest_house = np.where(datasetObj.flagSplit_house == 3)[0]
        houseIDList = datasetObj.houseIDList
        viewIDList = datasetObj.viewIDList
        indChosen = []
        numViewPerScene = 10
        for j_house in indTest_house.tolist():
            ind = np.where((houseIDList == j_house) & (flagSplit == 3))[0]
            assert np.all(viewIDList[ind[:-1]] < viewIDList[ind[1:]])
            nView = ind.shape[0]
            interval = int(nView / numViewPerScene)
            indChosenNow = ind[::interval][:numViewPerScene].tolist()
            assert len(indChosenNow) == numViewPerScene
            indChosen += indChosenNow
        assert np.all(flagSplit[indChosen] == 3)
        indVisChosen = indChosen[::10][:10]
        entry = dict(
            testDataNickName='scannetOfficialTestSplit%d' % numViewPerScene,
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indTest_house, houseIDList
        del viewIDList, indChosen, numViewPerScene, ind, nView, interval, indChosenNow, indVisChosen, entry

    # --------------------- submissionTimeScannetOfficialTestSplit ------------------ #
    if (wishedTestDataNickName is None) or \
            any([_.startswith('submissionTimeScannetOfficialTestSplit') for _ in wishedTestDataNickName]):
        meta = {}
        meta['scannetMeshCache'] = SubmissionTimeScannetMeshCache()
        datasetConf = {}
        datasetConf['did'] = 0
        datasetConf['dataset'] = 'scannetGivenRender'
        datasetConf['winWidth'] = 256
        datasetConf['winHeight'] = 192
        datasetConf['focalLengthWidth'] = 230.4
        datasetConf['focalLengthHeight'] = 230.4
        datasetConf['fScaleWidth'] = 2. * datasetConf['focalLengthWidth'] / datasetConf['winWidth']
        datasetConf['fScaleHeight'] = 2. * datasetConf['focalLengthHeight'] / datasetConf['winHeight']
        datasetConf['zNear'] = 1.e-6
        datasetObj = SubmissionTimeScannetGivenRenderDataset(
            datasetConf, projRoot=projRoot, datasetSplit='test',
            R='RlocalNono',  # read from /shared/group as usual
        )
        flagSplit = datasetObj.flagSplit
        indTest_house = np.where(datasetObj.flagSplit_house == 3)[0]
        houseIDList = datasetObj.houseIDList
        viewIDList = datasetObj.viewIDList
        indChosen = []
        numViewPerScene = 10
        for j_house in indTest_house.tolist():
            ind = np.where((houseIDList == j_house) & (flagSplit == 3))[0]
            assert np.all(viewIDList[ind[:-1]] < viewIDList[ind[1:]])
            nView = ind.shape[0]
            interval = int(nView / numViewPerScene)
            indChosenNow = ind[::interval][:numViewPerScene].tolist()
            assert len(indChosenNow) == numViewPerScene
            indChosen += indChosenNow
        assert np.all(flagSplit[indChosen] == 3)
        indVisChosen = indChosen[::10][:10]
        entry = dict(
            testDataNickName='submissionTimeScannetOfficialTestSplit%d' % numViewPerScene,
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indTest_house, houseIDList
        del viewIDList, indChosen, numViewPerScene, ind, nView, interval, indChosenNow, indVisChosen, entry

    # ------------------- freeDemoRender -------------------- #
    for testDataNickName in ['freedemo1']:
        if (wishedTestDataNickName is None) or (testDataNickName in wishedTestDataNickName):
            meta = {}
            datasetConf = {}
            datasetConf['did'] = 0
            datasetConf['dataset'] = testDataNickName
            datasetConf['winWidth'] = 256
            datasetConf['winHeight'] = 192
            datasetConf['focalLengthWidth'] = 230.4
            datasetConf['focalLengthHeight'] = 230.4
            datasetConf['fScaleWidth'] = 2. * datasetConf['focalLengthWidth'] / datasetConf['winWidth']
            datasetConf['fScaleHeight'] = 2. * datasetConf['focalLengthHeight'] / datasetConf['winHeight']
            datasetConf['zNear'] = 1.e-6
            datasetObj = FreeDemoRenderDataset(datasetConf, projRoot=projRoot)
            entry = dict(
                testDataNickName=testDataNickName,
                meta=meta, datasetObj=datasetObj,
                indChosen=list(range(len(datasetObj))),
                indVisChosen=list(range(len(datasetObj))),
            )
            testDataEntryDict[testDataNickName] = entry
            del meta, datasetConf, datasetObj, entry

    # ------------------- More to Come -------------------- #

    return testDataEntryDict
