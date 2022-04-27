# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configs_registration import getConfigGlobal
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..approachEntryPool import getApproachEntryDict
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import mkdir_full, tabularPrintingConstructing, tabularCvsDumping
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from collections import OrderedDict
import pickle
import os
import sys
import numpy as np


bt = lambda s: s[0].upper() + s[1:]
governedScriptTagList = ['1E', '1Ed', '1G']


# =================================== Common Tools ==================================== #


def abTheMeshInTheWorldSys(xyzWorld0, camR0, camT0, fScaleWidth, fScaleHeight, a, b, zNear):
    assert len(xyzWorld0.shape) == 2 and xyzWorld0.shape[1] == 3
    assert camR0.shape == (3, 3)
    assert camT0.shape == (3, )
    assert type(fScaleWidth) is float
    assert type(fScaleHeight) is float
    assert type(a) is float
    assert type(b) is float
    assert type(zNear) is float
    xyzCam0 = np.matmul(xyzWorld0, camR0.transpose()) + camT0[None, :]
    xyzCamPersp0 = np.stack([
        xyzCam0[:, 0] * fScaleWidth / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 1] * fScaleHeight / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 2],
    ], 1)
    xyzCamPersp0[:, 2] = a * xyzCamPersp0[:, 2] + b
    xyzCam0_new = np.stack([
        xyzCamPersp0[:, 0] * xyzCamPersp0[:, 2] / fScaleWidth,
        xyzCamPersp0[:, 1] * xyzCamPersp0[:, 2] / fScaleHeight,
        xyzCamPersp0[:, 2]
    ], 1)
    xyzWorld0_new = np.matmul(xyzCam0_new - camT0[None, :], camR0)  # camR0: transpose of inverse
    return xyzWorld0_new


def abTheMeshInTheCamSys(xyzCam0, fScaleWidth, fScaleHeight, a, b, zNear):
    assert len(xyzCam0.shape) == 2 and xyzCam0.shape[1] == 3
    assert type(fScaleWidth) is float
    assert type(fScaleHeight) is float
    assert type(a) is float
    assert type(b) is float
    assert type(zNear) is float
    xyzCamPersp0 = np.stack([
        xyzCam0[:, 0] * fScaleWidth / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 1] * fScaleHeight / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 2],
    ], 1)
    xyzCamPersp0[:, 2] = a * xyzCamPersp0[:, 2] + b
    xyzCam0_new = np.stack([
        xyzCamPersp0[:, 0] * xyzCamPersp0[:, 2] / fScaleWidth,
        xyzCamPersp0[:, 1] * xyzCamPersp0[:, 2] / fScaleHeight,
        xyzCamPersp0[:, 2]
    ], 1)
    return xyzCam0_new


def judgeWhichTrianglesFlagIn(
        vertCam0, face0, camR0, camT0, fScaleWidth, fScaleHeight, zNear,
        tightened, boundMaxWorld0, boundMinWorld0):
    vertWorld0 = np.matmul(vertCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse
    if face0.shape[0] == 0:
        vertCam0 = np.eye(3).astype(np.float32)
        face0 = np.array([[0, 1, 2]], dtype=np.int32)
    vertCamPersp0 = np.stack([
        fScaleWidth * vertCam0[:, 0] /
        np.clip(vertCam0[:, 2], a_min=zNear, a_max=np.inf),
        fScaleHeight * vertCam0[:, 1] /
        np.clip(vertCam0[:, 2], a_min=zNear, a_max=np.inf),
        vertCam0[:, 2],
    ], 1)
    faceVertCamPersp0 = vertInfo2faceVertInfoNP(
        vertCamPersp0[None], face0[None])[0]
    faceExtremeCamPersp0 = np.abs(faceVertCamPersp0).max(1)
    faceVertWorld0 = vertInfo2faceVertInfoNP(
        vertWorld0[None], face0[None])[0]
    faceMaxWorld0 = faceVertWorld0.max(1)
    faceMinWorld0 = faceVertWorld0.min(1)
    flagIn = (
            (np.abs(faceExtremeCamPersp0[:, 0]) < tightened) &
            (np.abs(faceExtremeCamPersp0[:, 1]) < tightened)
    )
    for xyz in range(3):
        flagIn &= (
                (faceMaxWorld0[:, xyz] < boundMaxWorld0[xyz]) &
                (faceMinWorld0[:, xyz] > boundMinWorld0[xyz])
        )
    return flagIn


# =================================== Benchmarking ==================================== #


def _wrong():
    print('Something is wrong. Please check!')
    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def assertBenchmarkings(bsv0_toStore, bsv0_loaded, **kwargs):
    verbose = kwargs['verbose']
    approachNickName = kwargs['approachNickName']

    # general must be exactly the same
    for k in ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
              'index', 'did', 'datasetID', 'dataset', 'flagSplit']:
        if not bsv0_loaded[k] == bsv0_toStore[k]:
            _wrong()
    thre_strict = 1.e-6  # if not approachNickName.startswith('P16') else 1.e-3
    # for k in ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete']:
    for k in ['MetricR1', 'FitR1']:  # , 'MetricR1ViaD', 'MetricF1ViaD', 'FitR1ViaD']:
        S = float(bsv0_toStore['finalBen%s' % bt(k)])
        L = float(bsv0_loaded['finalBen%s' % bt(k)])
        delta = abs((S - L) / (L + 1.e-5))
        if verbose:
            print('%s: %f' % (k, delta))
        if not delta < thre_strict:
            _wrong()
    thre_lenient = 1.e-4  # if not approachNickName.startswith('P16') else 1.e-3
    # for k in ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']:
    for k in ['MetricR1', 'MetricF1', 'MetricChamfer', 'FitR1']:
        S = float(bsv0_toStore['finalBen%s' % bt(k)])
        L = float(bsv0_loaded['finalBen%s' % bt(k)])
        delta = abs((S - L) / (L + 1.e-5))
        if np.isnan(S) and np.isnan(L):
            delta = 0
        if verbose:
            print('%s: %f' % (k, delta))
        if not delta < thre_lenient:
            _wrong()


def get_s(bsv0_retrieveList_dict, approachEntryDict, testDataEntry, **kwargs):
    # bsv0_retrieveList_dict[approachNickName][index] is bsv0

    # from testDataEntry
    indChosen = testDataEntry['indChosen']

    s = OrderedDict([])
    for approachNickName in bsv0_retrieveList_dict.keys():
        approachShownName = approachEntryDict[approachNickName]['approachShownName']
        s[approachNickName] = OrderedDict([])
        s[approachNickName]['method'] = approachShownName

        record = {}

        benPool = []
        for metricOrFit in ['Fit', 'Metric']:
            for name in ['Acc', 'Compl', 'Chamfer', 'Prec', 'Recall', 'F1',
                         'AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE',
                         'R1', 'R2', 'R3', 'Complete']:
                benPool.append(metricOrFit + name)

        for k in benPool:
            record[k] = []

        for j in indChosen:
            bsv0 = bsv0_retrieveList_dict[approachNickName][j]
            for k in benPool:
                if 'finalBen%s' % bt(k) in bsv0.keys():
                    record[k].append(bsv0['finalBen%s' % bt(k)])

        for k in benPool:
            if len(record[k]) == 0:
                s[approachNickName][k] = '-'
                continue
            t = np.array(record[k], dtype=np.float32)

            # assert np.isnan(t).sum() < 10
            if np.isnan(t).sum() >= 15:
                print('Nan cases are more than 15! (%d) for %s. Please Check.' %
                      (np.isnan(t).sum(), approachNickName))
                import ipdb
                ipdb.set_trace()
                print(1 + 1)

            t = np.nanmean(t)
            if k in []:
                s[approachNickName][k] = '%.3f' % t
            else:
                s[approachNickName][k] = '%.1f' % (t * 100.)
    return s


def main():
    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    testDataEntryDict = getTestDataEntryDict()
    approachEntryDict = getApproachEntryDict()
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    # set your set - testDataEntryKey
    testDataNickName = 'scannetOfficialTestSplit10'
    testDataEntry = testDataEntryDict[testDataNickName]

    # from testDataEntry
    datasetObj = testDataEntry['datasetObj']
    dataset = datasetObj.datasetConf['dataset']
    indChosen = testDataEntry['indChosen']
    flagSplit = datasetObj.flagSplit

    # run over all the approaches
    bsv0_retrieveList_dict = OrderedDict([])
    for approachNickName in approachEntryDict.keys():
        approachEntry = approachEntryDict[approachNickName]

        # from approachEntry
        approachNickName, methodologyName, scriptTag, approachShownName = \
            approachEntry['approachNickName'], approachEntry['methodologyName'], \
            approachEntry['scriptTag'], approachEntry['approachShownName']

        if scriptTag in governedScriptTagList:
            print('Dumping the Csv File. Currently Gathering Approach %s' % approachNickName)

            # assertions and mkdir
            outputBen_root = Btag_root + 'ben/%s/%s_%s/' % \
                             (scriptTag, testDataNickName, approachNickName)
            assert os.path.isdir(outputBen_root), outputBen_root

            # record
            bsv0_retrieveList_dict[approachNickName] = OrderedDict([])
            count = 0
            for j in indChosen:
                outputBenFileName = outputBen_root + '%08d.pkl' % j
                if not os.path.isfile(outputBenFileName):
                    print('Problem occurred. File not exist for %s. Please check.' % outputBenFileName)
                    import ipdb
                    ipdb.set_trace()
                    raise ValueError('You cannot proceed.')
                else:
                    with open(outputBenFileName, 'rb') as f:
                        bsv0 = pickle.load(f)
                    bsv0_retrieveList_dict[approachNickName][j] = bsv0
                count += 1
    s = get_s(
        bsv0_retrieveList_dict,
        approachEntryDict,
        testDataEntry,
    )
    print(tabularPrintingConstructing(
        s,
        field_names=list(list(s.values())[0].keys()),
        ifNeedToAddTs1=False,
    ))
    mkdir_full(projRoot + 'cache/B_%s/' % Btag)
    csv_fn = projRoot + 'cache/B_%s/benchmarking_%s_%s.csv' % \
             (Btag, Btag, testDataNickName)
    tabularCvsDumping(
        fn=csv_fn,
        s=s,
        fieldnames=list(list(s.values())[0].keys()),
        ifNeedToAddTs1=False,
    )


if __name__ == '__main__':
    main()