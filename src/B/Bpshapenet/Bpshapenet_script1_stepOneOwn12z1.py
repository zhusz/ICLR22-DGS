# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
import pickle
import copy
from configs_registration import getConfigGlobal
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_framework.framework_util_v2 import castAnything
import numpy as np


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    sys.path.append(projRoot + 'src/P/')

    # tags
    Btag = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    stepTag = 'stepOneOwn12z1'

    # user control
    methodologyChosen = os.environ['MC']

    # dataset and indChosen
    dataset = 'corenetChoySingleRendering'
    A1 = udl('pkl_A1b_', dataset)
    flagSplit = A1['flagSplit']
    m = A1['m']
    catNameList = A1['catNameList']
    del A1
    indChosen = np.where(flagSplit == 3)[0].tolist()
    indChosen = indChosen[:1898]

    # methodology list
    # (methdologyName (e.g. PDSRI), testSuiteName (e.g. PDClass), methodologyNickName)
    methodologyTupList = [
        ('PshapenetD0corenetdgsS2RreleaseIlatest', None, '(Ours) DGS', 'stepOneOwn12z1'),
        ('PshapenetD0corenetS0RreleaseIlatest', None, '(Oracle) CoReNet', 'stepOneOwn12z1'),
        ('PshapenetD0disndvrdgsS2RreleaseIlatest', None, '(Ours) DGS Best', 'stepOneOwn12z1'),
        ('PshapenetD0disndvrS2RreleaseIlatest', None, '(Oracle) DISN DVR', 'stepOneOwn12z1'),
    ]

    # grands
    pass

    # assertions
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    # mkdir
    visualDir = projRoot + 'cache/B_%s_%s/%s/' % (Btag, dataset, stepTag)
    mkdir_full(visualDir)

    # stat to record
    pass

    for methodologyTup in methodologyTupList:
        # basic info
        methodologyName, testSuiteName, methodologyNickName, _ = methodologyTup
        methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = \
            splitPDSRI(methodologyName)
        if not testSuiteName:
            testSuiteName = methodologyP + methodologyD + 'DVisualizer'
        testSuiteP, testSuiteD, testSuiteVisualizerClassName = splitPDDrandom(testSuiteName)

        # whether to run?
        if methodologyName != methodologyChosen and methodologyChosen != 'all':
            continue

        # mkdir
        output_root = Btag_root + '%s/%s_%s_%s/' % (stepTag, testSuiteName, dataset, methodologyName)
        mkdir_full(output_root)
        visualDir = projRoot + 'cache/B_%s/dummy/' % dataset  # This script does not produce any visuals
        mkdir_full(visualDir)

        # datasetObj
        DDataset = getConfigGlobal(
            methodologyP, methodologyD, methodologyS, methodologyR,
            wishedClassNameList=None,
        )['exportedClasses']['DCorenetChoySingleRenderingDataset']
        datasetMetaConf = {
            # winSize
            'winHeight': 256,
            'winWidth': 256,
        }
        datasetConf = copy.deepcopy(datasetMetaConf)
        datasetConf['did'] = 0
        datasetConf['dataset'] = 'corenetChoySingleRendering'
        datasetConf['batchSize'] = 1
        datasetConf['trainingSplit'] = 'test'
        datasetObj = DDataset(
            datasetConf, projRoot=projRoot,
            datasetSplit='test',
            cudaDevice='cuda:0',
        )

        # save to grand - datasetObj
        pass

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

        # visualizer
        DVisualizer = getConfigGlobal(
            testSuiteP, testSuiteD, 'SNono', 'RNono',
            wishedClassNameList=testSuiteVisualizerClassName,
        )['exportedClasses'][testSuiteVisualizerClassName]
        visualizer = DVisualizer(visualDir=visualDir, ifTracked=True)
        visualizer.setVisualMeta()

        # save to grand - visualizer
        pass

        # counter
        count = 0

        for j in indChosen:
            outputFileName = output_root + '%08d.pkl' % j

            if os.path.exists(outputFileName):
                print('Reading %s (flagSplit %d) - count (%d / %d)' % (
                    outputFileName, flagSplit[j], count, len(indChosen)
                ))
                # This script has no stat to record, so we even do not bother to read the pkl
            else:
                print('Processing %s (flagSplit %d) - count (%d / %d)' % (
                    outputFileName, flagSplit[j], count, len(indChosen)
                ))

                batch0_np = datasetObj.getOneNP(j)
                batch_np = {k: batch0_np[k][None] for k in batch0_np.keys()}
                batch_vis = batch_np
                bsv0_initial = visualizer.constructInitialBatchStepVis0(
                    batch_vis, iterCount=trainer.resumeIter, visIndex=0,
                    P=methodologyP, D=methodologyD, S=methodologyS, R=methodologyR,
                    verboseGeneral=visualizer.verboseGeneralBp12,
                )
                bsv0_initial = visualizer.mergeFromBatchVis(bsv0_initial, batch_vis)
                bsv0_stepOne = visualizer.stepOneBp12(
                    bsv0_initial, meta=trainer.meta, models=trainer.models, datasets=[datasetObj],
                    generalVoxelizationFunc=trainer._generalVoxelization,
                    samplingFromNetwork1ToNetwork2Func=trainer._samplingFromNetwork1ToNetwork2,
                    samplingFromNetwork2ToNetwork3Func=trainer._samplingFromNetwork2ToNetwork3,
                )
                bsv0_stepOne_toStore = {
                    k: bsv0_stepOne[k] for k in bsv0_stepOne.keys()
                    if k in ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                             'index', 'did', 'datasetID', 'dataset', 'flagSplit',
                             'corenetCamcubePredBisemYXZPacked', 'img',
                             'corenetCamcubeLabelBisemYXZPacked',
                             'corenetPredVertCam', 'corenetPredFace', 'corenetCubeIou']
                }
                with open(outputFileName, 'wb') as f:
                    pickle.dump(bsv0_stepOne_toStore, f)

            count += 1


if __name__ == '__main__':
    main()


