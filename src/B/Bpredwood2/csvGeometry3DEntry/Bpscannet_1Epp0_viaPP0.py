# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configs_registration import getConfigGlobal
from .dumpCsvGeometry3D import get_s, assertBenchmarkings
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..approachEntryPool import getApproachEntryDict
from codes_py.toolbox_framework.framework_util_v4 import splitPDSRI, splitPDDrandom, \
    constructInitialBatchStepVis0, mergeFromBatchVis, bsv02bsv
from codes_py.py_ext.misc_v1 import tabularPrintingConstructing, tabularCvsDumping
from collections import OrderedDict
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
import copy
import pickle
import os
import sys
import numpy as np


from .benchmarkingGeometry3DScannet import benchmarkingGeometry3DScannetFunc
from .benchmarkingGeometry3DDemo import benchmarkingGeometry3DDemoFunc
from .benchmarkingGeometry3DHm import benchmarkingGeometry3DHmFunc


# easy debugging
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPlyPointCloud
from scipy.io import savemat


bt = lambda s: s[0].upper() + s[1:]


def testApproachOnTestData(testDataEntry, approachEntry):
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    sys.path.append(projRoot + 'src/P/')
    ifDebug = 1 if ('DB' in os.environ.keys() and os.environ['DB'] == '1') else 0

    # from testDataEntry
    datasetObj = testDataEntry['datasetObj']
    indChosen = testDataEntry['indChosen']
    indVisChosen = testDataEntry['indVisChosen']
    testDataNickName = testDataEntry['testDataNickName']
    flagSplit = datasetObj.flagSplit
    datasetMeta = testDataEntry['meta']
    if testDataNickName in ['scannetOfficialTestSplit10', 'freedemo1']:
        pyrenderManager = PyrenderManager(256, 192)
    elif testDataNickName in [
            'hmOfficialValSplit10', 'hmFsOfficialValSplit10',
            'freedemo1', 'freedemo2', 'freedemo3', 'freedemo11',
            'squareFreedemo1', 'squareFreedemo2', 'squareFreedemo3', 'squareFreedemo11',
        ]:
        pyrenderManager = PyrenderManager(256, 256)
    else:
        raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)

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
    trainer.initializeAll(iter_label=int(methodologyI[1:]), hook_type=None)
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
            if testDataNickName in ['hmOfficialValSplit10', 'hmFsOfficialValSplit10']:
                bsv0 = benchmarkingGeometry3DHmFunc(
                    bsv0_initial,
                    hmMeshCache=datasetMeta['hmMeshCache'],
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice='cuda:0',
                    raiseOrTrace='trace' if ifDebug else 'ignoreAndNan',
                    # benchmarking rules
                    voxSize=128, numMeshSamplingPoint=200000, # If you wish to test with resolution 256, you need to create a new script called 1E256
                    # drawing
                    ifRequiresDrawing=(j in indVisChosen),
                    pyrenderManager=pyrenderManager,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=trainer.models, verboseBatchForwarding=0,
                    Trainer=type(trainer), doPred0Func=trainer.doPred0,
                    useSurfaceOrDepth='postprocess0', meta=trainer.meta,
                )
            elif testDataNickName in ['scannetOfficialTestSplit10']:
                bsv0 = benchmarkingGeometry3DScannetFunc(
                    bsv0_initial,
                    scannetMeshCache=datasetMeta['scannetMeshCache'],
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice='cuda:0',
                    raiseOrTrace='trace' if ifDebug else 'ignoreAndNan',
                    # benchmarking rules
                    voxSize=128, numMeshSamplingPoint=200000,  # If you wish to test with resolution 256, you need to create a new script called 1E256
                    # drawing
                    ifRequiresDrawing=(j in indVisChosen),
                    pyrenderManager=pyrenderManager,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=trainer.models, verboseBatchForwarding=0,
                    Trainer=type(trainer), doPred0Func=trainer.doPred0,
                    useSurfaceOrDepth='postprocess0', meta=trainer.meta,
                )
            elif testDataNickName in [
                        'demo1', 'freedemo1', 'pix3d', 'freedemo2', 'freedemo3',
                        'squareFreedemo1', 'squareFreedemo2', 'squareFreedemo3', 'squareFreedemo11',
                    ]:
                bsv0 = benchmarkingGeometry3DDemoFunc(
                    bsv0_initial,
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice='cuda:0',
                    raiseOrTrace='trace' if ifDebug else 'ignoreAndNan',
                    # benchmarking rules
                    voxSize=128, numMeshSamplingPoint=200000,
                    # drawing
                    ifRequiresDrawing=(j in indVisChosen),
                    pyrenderManager=pyrenderManager,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=trainer.models, verboseBatchForwarding=0,
                    Trainer=type(trainer), doPred0Func=trainer.doPred0,
                    useSurfaceOrDepth='postprocess0', meta=trainer.meta,
                )
            else:
                raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)

            # outputBenFile
            bsv0_toStore = {
                k: bsv0[k] for k in bsv0.keys()
                if (k in ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                          'index', 'did', 'datasetID', 'dataset', 'flagSplit']
                    or ('finalBen' in k))
            }
            if os.path.exists(outputBenFileName):
                with open(outputBenFileName, 'rb') as f:
                    bsv0_loaded = pickle.load(f)
                if testDataNickName in ['scannetOfficialTestSplit10', 'hmFsOfficialValSplit10']:
                    assertBenchmarkings(
                        bsv0_toStore, bsv0_loaded, verbose=False,
                        approachNickName=approachNickName)
                elif testDataNickName in [
                            'demo1', 'freedemo1', 'pix3d', 'freedemo2', 'freedemo3',
                            'squareFreedemo1', 'squareFreedemo2', 'squareFreedemo3', 'squareFreedemo11',
                        ]:
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
                        'houseID', 'viewID',
                        'imgForUse', 'depthForUse', 'imgRaw', 'depthRaw', 'predDepth2',
                        'EWorld', 'LWorld', 'UWorld', 'cam',
                    ]) or ('MeshDrawingPackage' in k) or ('finalBen' in k) 
                    or ('MetricVert' in k) or ('FitVert' in k)  # do not dump faceVert / faceCentroid...
                    or ('MetricFace' in k) or ('FitFace' in k)
                    or ('depthRenderedFrom' in k)
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
