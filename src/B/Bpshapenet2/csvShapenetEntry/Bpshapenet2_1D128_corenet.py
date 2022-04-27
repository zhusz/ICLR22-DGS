# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configs_registration import getConfigGlobal
from .dumpCsvShapenet import get_s, assertBenchmarkings
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..approachEntryPool import getApproachEntryDict
from codes_py.toolbox_framework.framework_util_v4 import splitPDSRI, splitPDDrandom, \
    constructInitialBatchStepVis0, mergeFromBatchVis, bsv02bsv
from codes_py.py_ext.misc_v1 import tabularPrintingConstructing, tabularCvsDumping
from collections import OrderedDict
import copy
import pickle
import os
import sys
import numpy as np


from .benchmarkingShapenetCorenet import benchmarkingShapenetCorenetFunc


def testApproachOnTestData(testDataEntry, approachEntry):
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    sys.path.append(projRoot + 'src/P/')

    # from testDataEntry
    datasetObj = testDataEntry['datasetObj']
    indChosen = testDataEntry['indChosen']
    indVisChosen = testDataEntry['indVisChosen']
    testDataNickName = testDataEntry['testDataNickName']
    flagSplit = datasetObj.flagSplit
    datasetMeta = testDataEntry['meta']

    # from approachEntry
    approachNickName, methodologyName, scriptTag = \
        approachEntry['approachNickName'], \
        approachEntry['methodologyName'], approachEntry['scriptTag']
    methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = \
        splitPDSRI(methodologyName)

    # assertions and mkdir
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)
    outputBen_root = Btag_root + 'ben/%s/%s_%s/' % \
                     (scriptTag, testDataNickName, approachNickName)
    os.makedirs(outputBen_root, exist_ok=True)
    outputVis_root = Btag_root + 'vis/%s/%s_%s/' % \
                     (scriptTag, testDataNickName, approachNickName)
    os.makedirs(outputVis_root, exist_ok=True)

    # trainer
    DTrainer = getConfigGlobal(
        methodologyP, methodologyD, methodologyS, methodologyR,
        wishedClassNameList=['DTrainer'],
    )['exportedClasses']['DTrainer']
    getConfigFunc = getConfigGlobal(
        methodologyP, methodologyD, methodologyS, methodologyR,
        wishedClassNameList=[],
    )['getConfigFunc']
    config = getConfigFunc(methodologyP, methodologyD, methodologyS, methodologyR)
    trainer = DTrainer(config, rank=0, numMpProcess=0, ifDumpLogging=False)
    trainer.initializeAll(iter_label=methodologyI[1:], hook_type=None)
    trainer.setModelsEvalMode(trainer.models)

    # record
    bsv0_retrieveList_dict = OrderedDict([])
    bsv0_retrieveList_dict[approachNickName] = OrderedDict([])

    j1 = int(os.environ.get('J1', -1))
    j2 = int(os.environ.get('J2', -1))

    # counter
    if (j1 >= 0) and (j2 >= 0):
        count = j1
    else:
        count = 0

    for i, j in enumerate(indChosen):
        if (j1 >= 0) and (j2 >= 0) and ((i < j1) or (i >= j2)):
            continue

        outputBenFileName = outputBen_root + '%08d.pkl' % j
        outputVisFileName = outputVis_root + '%08d.pkl' % j

        if os.path.exists(outputBenFileName) and \
                ((j not in indVisChosen) or (os.path.exists(outputVisFileName))):
            print('Reading %s (flagSplit %d) - count (%d / %d)' % (
                outputBenFileName, flagSplit[j], count, len(indChosen)
            ))

            with open(outputBenFileName, 'rb') as f:
                bsv0_toStore = pickle.load(f)
        else:
            print('Processing %s (flagSplit %d) - count (%d / %d)' % (
                outputBenFileName, flagSplit[j], count, len(indChosen)
            ))

            batch0_np = datasetObj.getOneNP(j)
            batch_np = bsv02bsv(batch0_np)
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis, iterCount=trainer.resumeIter, visIndex=0,
                dataset=None,  # only during training you need to input the dataset here.
                P=methodologyP, D=methodologyD, S=methodologyS, R=methodologyR,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromBatchVis(
                bsv0_initial, batch_vis, dataset=None, visIndex=0)

            # step
            if testDataNickName in [
                'corenetSingleOfficialTestSplitFirstOnePerCent',
                'corenetSingleOfficialTestSplit',
            ]:
                bsv0 = benchmarkingShapenetCorenetFunc(
                    bsv0_initial,
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice='cuda:0',
                    # benchmarking rules
                    camcubeVoxSize=128,  # sb
                    # drawing
                    ifRequiresDrawing=(j in indVisChosen),
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=trainer.models, verboseBatchForwarding=0,
                    Trainer=type(trainer), doPred0Func=trainer.doPred0,
                    meta=trainer.meta,
                )
            else:
                raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)

            # outputBenFile
            bsv0_toStore = {
                k: bsv0[k] for k in bsv0.keys()
                if (k in ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                          'index', 'did', 'datasetID', 'dataset', 'flagSplit']
                    or ('finalBen' in k)) or (k == 'catName')
            }
            if os.path.exists(outputBenFileName):
                with open(outputBenFileName, 'rb') as f:
                    bsv0_loaded = pickle.load(f)
                if testDataNickName in [
                    'corenetSingleOfficialTestSplitFirstOnePerCent',
                    'corenetSingleOfficialTestSplit',
                ]:
                    assertBenchmarkings(
                        bsv0_toStore, bsv0_loaded, verbose=False,
                        approachNickName=approachNickName)
                elif testDataNickName in []:
                    pass
                else:
                    raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)
            else:
                with open(outputBenFileName, 'wb') as f:
                    pickle.dump(bsv0_toStore, f)

            # outputVisFile
            if (j in indVisChosen) and (not os.path.exists(outputVisFileName)):
                bsv0_forVis = {
                    k: bsv0[k] for k in bsv0.keys()
                    if (k in [
                        'iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                        'index', 'did', 'datasetID', 'dataset', 'flagSplit',
                        'img',
                        'rawVertCam', 'rawFace', 'rawVertRgb',
                        'corenetPredVertCam', 'corenetPredFace', 'corenetVertRgb',
                        'corenetLabelVertCam', 'corenetLabelFace', 'corenetVertRgb',
                        'corenetCamcubeLabelOccfloatYXZ', 'corenetCamcubePredOccfloatYXZ',
                        'corenetCamcubeMaskfloat',
                    ]) or ('MeshDrawingPackage' in k) or ('finalBen' in k)
                }
                with open(outputVisFileName, 'wb') as f:
                    pickle.dump(bsv0_forVis, f)

        bsv0_retrieveList_dict[approachNickName][j] = bsv0_toStore

        count += 1

    if (j1 < 0) and (j2 < 0):
        s = get_s(
            bsv0_retrieveList_dict,
            {approachEntry['approachNickName']: approachEntry},
            testDataEntry)
        print(tabularPrintingConstructing(
            s,
            field_names=list(list(s.values())[0].keys()),
            ifNeedToAddTs1=False,
        ))

