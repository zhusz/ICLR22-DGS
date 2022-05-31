# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

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
from .demoRenderDataset import DemoRenderDataset, FreeDemoRenderDataset
from .hmRenderOmDataset import HmRenderOmDataset, HmMeshCache


bt = lambda s: s[0].upper() + s[1:]


def getTestDataEntryDict(**kwargs):
    wishedTestDataNickName = kwargs['wishedTestDataNickName']
    assert type(wishedTestDataNickName) is list or wishedTestDataNickName is None

    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    sys.path.append(projRoot + 'src/P/')
    testDataEntryDict = {}

    # --------------------- hmFsOfficialValSplit10 ------------------ #
    if (wishedTestDataNickName is None) or \
            any([_.startswith('hmFsOfficialValSplit')] for _ in wishedTestDataNickName):
        meta = {}
        meta['hmMeshCache'] = HmMeshCache(zNear=1.e-6)
        datasetConf = {}
        datasetConf['dataset'] = 'hmRenderOm'
        datasetConf['winWidth'] = 256
        datasetConf['winHeight'] = 256
        datasetConf['zNear'] = 1.e-6
        datasetConf['ifNeedPixelAugmentation'] = 0
        datasetConf['ifNeedCroppingAugmentation'] = 1
        datasetConf['fScaleWidthDesiredFixed'] = 1.8  # the same as scannetGivenRender
        datasetConf['fScaleHeightDesiredFixed'] = 1.8
        datasetConf['ifNeedMirrorAugmentation'] = 0
        datasetConf['ifDummy'] = False
        datasetObj = HmRenderOmDataset(
            datasetConf, projRoot=projRoot, datasetSplit='val',
        )
        flagSplit = datasetObj.flagSplit
        indVal_house = np.where(datasetObj.flagSplit_house == 2)[0]
        houseIDList = datasetObj.houseIDList
        pointIDList = datasetObj.pointIDList
        viewIDList = datasetObj.viewIDList
        indChosen = []
        numPointPerScene = 10
        for j_house in indVal_house.tolist():
            if j_house in [885, 886, 887, 888, 889, 895, 896, 897, 898]:  # the rgb rendering provided in omnidata is not meaningful
                continue
            ind = np.where((houseIDList == j_house) & (flagSplit == 2))[0]
            pointIDList_ind = pointIDList[ind]
            candidatePointIDs = np.sort(np.unique(pointIDList_ind))
            assert candidatePointIDs.shape[0] >= 10, (j_house, candidatePointIDs)  # select 10 points with their first view
            nPoint = candidatePointIDs.shape[0]
            interval = int(float(nPoint) / numPointPerScene)
            pointChosenNow = candidatePointIDs[::interval][:numPointPerScene].tolist()
            assert len(pointChosenNow) == numPointPerScene
            indChosenNow = []
            for p in pointChosenNow:
                t = int(np.where(pointIDList_ind == p)[0][0])
                indChosenNow.append(ind[t])
            indChosen += indChosenNow
        assert np.all(flagSplit[indChosen] == 2)
        indVisChosen = indChosen[::91]  # 0~910

        entry = dict(
            testDataNickName='hmFsOfficialValSplit%s' % numPointPerScene,
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indVal_house, houseIDList
        del pointIDList, viewIDList, pointIDList_ind, indChosen, numPointPerScene, ind
        del candidatePointIDs, nPoint, interval, pointChosenNow, indChosenNow, p, t
        del indVisChosen

    # --------------------- hmOfficialValSplit10 ------------------ #
    if (wishedTestDataNickName is None) or \
            any([_.startswith('hmOfficialValSplit')] for _ in wishedTestDataNickName):
        meta = {}
        meta['hmMeshCache'] = HmMeshCache(zNear=1.e-6)
        datasetConf = {}
        datasetConf['dataset'] = 'hmRenderOm'
        datasetConf['winWidth'] = 256
        datasetConf['winHeight'] = 256
        datasetConf['zNear'] = 1.e-6
        datasetConf['ifNeedPixelAugmentation'] = 0
        datasetConf['ifNeedCroppingAugmentation'] = 0
        datasetConf['ifNeedMirrorAugmentation'] = 0
        datasetObj = HmRenderOmDataset(
            datasetConf, projRoot=projRoot, datasetSplit='val',
        )
        flagSplit = datasetObj.flagSplit
        indVal_house = np.where(datasetObj.flagSplit_house == 2)[0]
        houseIDList = datasetObj.houseIDList
        pointIDList = datasetObj.pointIDList
        viewIDList = datasetObj.viewIDList
        indChosen = []
        numPointPerScene = 10
        for j_house in indVal_house.tolist():
            if j_house in [885, 886, 887, 888, 889, 895, 896, 897, 898]:
                continue
            ind = np.where((houseIDList == j_house) & (flagSplit == 2))[0]
            pointIDList_ind = pointIDList[ind]
            candidatePointIDs = np.sort(np.unique(pointIDList_ind))
            assert candidatePointIDs.shape[0] >= 10, (j_house, candidatePointIDs)  # select 10 points with their first view
            nPoint = candidatePointIDs.shape[0]
            interval = int(float(nPoint) / numPointPerScene)
            pointChosenNow = candidatePointIDs[::interval][:numPointPerScene].tolist()
            assert len(pointChosenNow) == numPointPerScene
            indChosenNow = []
            for p in pointChosenNow:
                t = int(np.where(pointIDList_ind == p)[0][0])
                indChosenNow.append(ind[t])
            indChosen += indChosenNow
        assert np.all(flagSplit[indChosen] == 2)
        indVisChosen = indChosen[:5]

        entry = dict(
            testDataNickName='hmOfficialValSplit%s' % numPointPerScene,
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indVal_house, houseIDList
        del pointIDList, viewIDList, pointIDList_ind, indChosen, numPointPerScene, ind
        del candidatePointIDs, nPoint, interval, pointChosenNow, indChosenNow, p, t
        del indVisChosen

    # --------------------- scannetOfficialTestSplit10 ------------------ #
    if (wishedTestDataNickName is None) or \
            any([_.startswith('scannetOfficialTestSplit') for _ in wishedTestDataNickName]):
        meta = {}
        meta['scannetMeshCache'] = ScannetMeshCache()
        datasetConf = {}
        # datasetConf['did'] = 0
        datasetConf['dataset'] = 'scannetGivenRender'
        datasetConf['winWidth'] = 256
        datasetConf['winHeight'] = 192
        datasetConf['focalLengthWidth'] = 230.4
        datasetConf['focalLengthHeight'] = 230.4
        datasetConf['fScaleWidth'] = 2. * datasetConf['focalLengthWidth'] / datasetConf['winWidth']
        datasetConf['fScaleHeight'] = 2. * datasetConf['focalLengthHeight'] / datasetConf['winHeight']
        datasetConf['zNear'] = 1.e-6
        datasetConf['ifDummy'] = False
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
        indVisChosen = indChosen[::10][:10] + [2667980]  # This is the only line that can be changed.
        # indVisChosen = indChosen[::10][:100]
        # indVisChosen = indChosen[1::2][:25]
        # indVisChosen = [2480280, 2514216, 2490701, 2519133, 2522469, 2532117, 2544190,
        #                 2682970, 2545678]
        entry = dict(
            testDataNickName='scannetOfficialTestSplit%d' % numViewPerScene,
            meta=meta, datasetObj=datasetObj,
            indChosen=indChosen, indVisChosen=indVisChosen,
        )
        testDataEntryDict[entry['testDataNickName']] = entry
        del meta, datasetConf, datasetObj, flagSplit, indTest_house, houseIDList
        del viewIDList, indChosen, numViewPerScene, ind, nView, interval, indChosenNow, indVisChosen, entry

    # ------------------- freeDemoRender -------------------- #
    for testDataNickName in ['freedemo1', 'freedemo2', 'freedemo3', 'freedemo11']:
        if (wishedTestDataNickName is None) or (testDataNickName in wishedTestDataNickName):
            meta = {}
            datasetConf = {}
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

    # ------------------- squareFreeDemoRender -------------------- #
    for testDataNickName in ['squareFreedemo1', 'squareFreedemo2', 'squareFreedemo3', 'squareFreedemo11']:
        if (wishedTestDataNickName is None) or (testDataNickName in wishedTestDataNickName):
            meta = {}
            datasetConf = {}
            datasetConf['dataset'] = testDataNickName
            datasetConf['winWidth'] = 256
            datasetConf['winHeight'] = 256
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
