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
from codes_py.py_ext.misc_v1 import mkdir_full
from configs_registration import getConfigGlobal
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import mkdir_full, tabularPrintingConstructing, tabularCvsDumping
from collections import OrderedDict
import numpy as np
import pickle
import copy


def get_s(bsv0_retrieveList_dict, methodology_dict, indChosen, **kwargs):
    allCatName = kwargs['allCatName']
    s = OrderedDict([])
    for methodologyName in bsv0_retrieveList_dict.keys():
        methodologyNickName = methodology_dict[methodologyName]['methodologyNickName']
        s[methodologyName] = OrderedDict([])
        s[methodologyName]['method'] = methodologyNickName

        catRecord = OrderedDict([])
        for catName in allCatName:
            catRecord[catName] = []
        for j in indChosen:
            bsv0 = bsv0_retrieveList_dict[methodologyName][j]
            catName = bsv0['catName']
            catRecord[catName].append(bsv0)
        record_for_mean = []
        for catName in allCatName:
            if len(catRecord[catName]) == 0:
                s[methodologyName][catName + 'Iou'] = '0'
            else:
                s[methodologyName][catName + 'Iou'] = \
                    '%.01f' % \
                    (100 *
                     (sum([ben['iou'] for ben in catRecord[catName]]) / len(catRecord[catName])))
                record_for_mean.append(sum([ben['iou'] for ben in catRecord[catName]]) / len(catRecord[catName]))
        s[methodologyName]['meanIou'] = '%.01f' % (100 * sum(record_for_mean) / len(record_for_mean))
    return s


def getCats():
    cats = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "cabinet": "02933112",
        "chair": "03001627",
        "bench": "02828884",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088",
    }
    cats = OrderedDict(cats)
    cats_inverse = {cats[k]: k for k in cats.keys()}

    cat_ids = list(cats.values())
    cat_ids_to_catOrder = OrderedDict([
        (cat_ids[c], c)
        for c in range(len(cat_ids))
    ])
    return cats, cats_inverse, cat_ids, cat_ids_to_catOrder


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    sys.path.append(projRoot + 'src/P/')

    # tags
    Btag = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    stepTag = 'stepBen'

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
    grandMethodologyName = methodologyTupList[0][0]
    grand_visualizer = None
    grand_datasetObj = None

    # assertions
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    # mkdir
    visualDir = projRoot + 'cache/B_%s_%s/%s/' % (Btag, dataset, stepTag)
    mkdir_full(visualDir)

    # stat to record
    bsv0_retrieveList_dict = OrderedDict([])  # Store methodology-index-dependent data, which is bsv0
    methodology_dict = OrderedDict([])  # Store index-independent data, e.g. methodologyNickName

    for methodologyTup in methodologyTupList:
        # basic info
        methodologyName, testSuiteName, methodologyNickName, inputStepTag = methodologyTup
        if not testSuiteName:
            methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = \
                splitPDSRI(methodologyName)
            testSuiteName = methodologyP + methodologyD + 'DVisualizer'
        testSuiteP, testSuiteD, testSuiteVisualizerClassName = splitPDDrandom(testSuiteName)

        # whether to run?
        if methodologyName != methodologyChosen and methodologyChosen != 'all' and \
                methodologyName != grandMethodologyName:
            continue

        # mkdir
        output_root = Btag_root + '%s/%s_%s_%s/' % (stepTag, testSuiteName, dataset, methodologyName)
        mkdir_full(output_root)
        input_root = Btag_root + '%s/%s_%s_%s/' % (inputStepTag, testSuiteName, dataset, methodologyName)
        assert os.path.isdir(input_root)

        # datasetObj
        if methodologyName.startswith('P'):
            methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = \
                splitPDSRI(methodologyName)
            DDataset = getConfigGlobal(
                methodologyP, methodologyD, methodologyS, methodologyR,
                wishedClassNameList=None,
            )['exportedClasses']['DCorenetChoySingleRenderingDataset']
            datasetMetaConf = {
                # winSize
                'winHeight': 256,
                'winWidth': 256,
            }
            # datasetMetaConf['numSamplingFinalGridpoint'] = datasetMetaConf['gridPointLength'] ** 3
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
        else:
            datasetObj = grand_datasetObj

        # trainer
        pass  # do nothing since this is step 2

        # datasetObj - save to grand
        if methodologyName == grandMethodologyName:
            grand_datasetObj = datasetObj

        # visualizer
        DVisualizer = getConfigGlobal(
            testSuiteP, testSuiteD, 'SNono', 'RNono', wishedClassNameList=testSuiteVisualizerClassName,
        )['exportedClasses'][testSuiteVisualizerClassName]
        visualizer = DVisualizer(visualDir=visualDir, ifTracked=True)
        visualizer.setVisualMeta()

        # visualizer - save to grand
        if methodologyName == grandMethodologyName:
            grand_visualizer = visualizer

        # whether to run?
        if methodologyName != methodologyChosen and methodologyChosen != 'all':
            continue

        # counter
        count = 0

        # stat to record
        bsv0_retrieveList_dict[methodologyName] = OrderedDict([])
        methodology_dict[methodologyName] = OrderedDict([])

        for j in indChosen:
            outputFileName = output_root + '%08d.pkl' % j
            inputFileName = input_root + '%08d.pkl' % j

            if os.path.exists(outputFileName):
                print('Reading %s (flagSplit %d) - count (%d / %d)' % (
                    outputFileName, flagSplit[j], count, len(indChosen)
                ))
                # read the pkl used for visualization
                with open(outputFileName, 'rb') as f:
                    batch_stepVis0_stepBenChamfer_toStore = pickle.load(f)

            else:
                print('Processing %s (flagSplit %d) - count (%d / %d)' % (
                    outputFileName, flagSplit[j], count, len(indChosen)
                ))

                try:
                    with open(inputFileName, 'rb') as f:
                        batch_stepVis0_stepOne = pickle.load(f)
                except:
                    print('Input File Not Found. Please Check.')
                    import ipdb
                    ipdb.set_trace()
                    raise Exception  # You cannot proceed

                # It is visualizer.stepBen's responsibility to load "label" directly
                # from the dataset
                # We assume this is not a time consuming process
                # If your "label" needs to be processed for a long period of time,
                # You should do it in "A" and save the "label" directly into "v".
                # Consequently, we do not include index_dict in non-html scripts like this one - 2A.
                batch_stepVis0_stepBenChamfer = visualizer.stepBenBp12(
                    batch_stepVis0_stepOne,
                    datasets=[datasetObj],
                )

                batch_stepVis0_stepBenChamfer_toStore = {
                    k: batch_stepVis0_stepBenChamfer[k]
                    for k in batch_stepVis0_stepBenChamfer.keys()
                    if k in ['iterCount', 'visIndex', 'P', 'D', 'S', 'R',
                             'methodologyName', 'index', 'did',
                             'datasetID', 'dataset', 'flagSplit',
                             # 'catID', 'catName', 'chamfer10000', 'iou10000']
                             'catID', 'catName', 'iou']
                }

                with open(outputFileName, 'wb') as f:
                    pickle.dump(batch_stepVis0_stepBenChamfer_toStore, f)

            # record
            bsv0_retrieveList_dict[methodologyName][j] = batch_stepVis0_stepBenChamfer_toStore

            # in time dump
            pass  # do nothing - no dump in here

            # counter
            count += 1

        # record
        methodology_dict[methodologyName]['methodologyNickName'] = methodologyNickName

        # tabulating
        # allCatName = list(udl('pkl_A3_catNamesToCatIDs', dataset).keys())
    allCatName = list(getCats()[0].keys())

    # checkResultStat(bsv0_retrieveList_dict, methodology_dict, indChosen, allCatName=allCatName)

    s = get_s(bsv0_retrieveList_dict, methodology_dict, indChosen, allCatName=allCatName)
    print(tabularPrintingConstructing(
        s,
        field_names=list(list(s.values())[0].keys()),
        ifNeedToAddTs1=False,
    ))
    if methodologyChosen == 'all':
        csv_fn = projRoot + 'cache/B_%s_%s/' % (Btag, dataset) + stepTag + '.csv'
        tabularCvsDumping(
            fn=csv_fn,
            s=s,
            fieldnames=list(list(s.values())[0].keys()),
            ifNeedToAddTs1=False,
        )
        print('Benchmarking result has already been saved to %s' % csv_fn)
    else:
        print('The task of running the chosen methodology %s for %s is finished!' % (
            methodologyChosen, stepTag,
        ))


if __name__ == '__main__':
    main()

