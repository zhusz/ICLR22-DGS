# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'src/versions/')
from configs_registration import getConfigGlobal
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..approachEntryPool import getApproachEntryDict
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import mkdir_full, tabularPrintingConstructing, tabularCvsDumping
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, drawDepthDeltaSign
from collections import OrderedDict
import pickle
import numpy as np


bt = lambda s: s[0].upper() + s[1:]


def addToSummary0Txt0BrInds(
        summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict, **kwargs):
    # Not Functional
    visualDir = kwargs['visualDir']
    testDataNickName = kwargs['testDataNickName']
    dataset = kwargs['dataset']

    for approachNickName in approachEntryDict.keys():
        if approachNickName not in bsv0_forVis_dict.keys():
            continue
        approachEntry = approachEntryDict[approachNickName]
        approachShownName = approachEntry['approachShownName']
        bsv0_forVis = bsv0_forVis_dict[approachNickName]

        for meshName in ['evalPred', 'evalView']:
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            if '%s%s%sMeshDrawingPackageNumView' % \
                    (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType)) \
                    not in bsv0_forVis.keys():
                continue

            # input image / depth delta sign visualization
            if meshName == 'evalView' or 'depthForUse' not in bsv0_forVis.keys():
                summary0['%s Input Img (%s)' % (approachShownName, meshName)] = \
                    bsv0_forVis['imgForUse'].transpose((1, 2, 0))
                txt0.append('')
            elif meshName == 'evalPred':
                summary0['%s delta sign' % approachShownName] = \
                    drawDepthDeltaSign(
                        bsv0_forVis['depthRenderedFromEvalPred2'],
                        bsv0_forVis['depthForUse'][0, :, :],
                        deltaThre=0.05,
                    )
                txt0.append('')
            else:
                raise NotImplementedError('Unknown meshName: %s' % meshName)

            # depth
            if meshName == 'evalPred':
                summary0['%s depthPred2' % approachShownName] = \
                    getImshow(bsv0_forVis['depthRenderedFromEvalPred2'],
                              vmin=None, vmax=None, cmap='inferno')
            elif meshName == 'evalView':
                summary0['%s depthLabel' % approachShownName] = \
                    getImshow(bsv0_forVis['depthForUse'][0, :, :],
                              vmin=None, vmax=None, cmap='inferno')
            else:
                raise NotImplementedError('Unknown meshName: %s' % meshName)
            if 'finalBenMetricR1' in bsv0_forVis.keys() and meshName == 'evalPred':
                txt0.append('AbsDiff: %.3f, AbsRel: %.3f, r1: %.3f' % (
                    bsv0_forVis['finalBenMetricAbsDiff'],
                    bsv0_forVis['finalBenMetricAbsRel'],
                    bsv0_forVis['finalBenMetricR1'],
                ))
            else:
                txt0.append('')

            # floor plan
            summary0['%s %s Floor Plan' % (approachShownName, meshName)] = \
                bsv0_forVis['%s%s%sMeshDrawingPackageFloorPlan' %
                            (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))]
            if 'finalBenMetricF1' in bsv0_forVis.keys() and meshName == 'evalPred':
                txt0.append('CD: %.3f, P: %.3f, R: %.3f, F1: %.3f' %
                            (bsv0_forVis['finalBenMetricChamfer'],
                             bsv0_forVis['finalBenMetricPrec'],
                             bsv0_forVis['finalBenMetricRecall'],
                             bsv0_forVis['finalBenMetricF1']))
            else:
                txt0.append('')

            # each view rendering
            numView = bsv0_forVis['%s%s%sMeshDrawingPackageNumView' %
                                  (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))]
            for v in range(numView):
                summary0['%s %s View %d' % (approachShownName, meshName, v)] = \
                    bsv0_forVis['%s%s%sMeshDrawingPackage%s' %
                                (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType),
                                 'ViewColor')][v, :, :, :]
                txt0.append('')
            brInds.append(len(summary0))

        # dump ply if you wish
        for meshName in ['evalPred', 'evalView']:
            sysLabel = 'cam'
            if '%sVertRgb' % meshName in bsv0_forVis.keys():
                dumpPly(
                    visualDir + '%s_%s_%d(%d)_mesh_%s_%s_%s.ply' %
                    (testDataNickName, dataset,
                     bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                     approachNickName,
                     meshName, sysLabel),
                    bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                    bsv0_forVis['%sFace' % meshName],
                    bsv0_forVis['%sVertRgb' % meshName],
                )
            if '%sFaceRgb' % meshName in bsv0_forVis.keys():
                dumpPly2(
                    visualDir + '%s_%s_%d(%d)_mesh_%s_%s_%s.ply' %
                    (testDataNickName, dataset,
                     bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                     approachNickName,
                     meshName, sysLabel),
                    bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                    bsv0_forVis['%sFace' % meshName],
                    bsv0_forVis['%sFaceRgb' % meshName],
                )
            if ('%sVert%s' % (meshName, bt(sysLabel)) in bsv0_forVis.keys()) and \
                    (not ('%sVertRgb' % meshName in bsv0_forVis.keys())):
                dumpPly(
                    visualDir + '%s_%s_%d(%d)_mesh_%s_%s_%s.ply' %
                    (testDataNickName, dataset,
                     bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                     approachNickName,
                     meshName, sysLabel),
                    bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                    bsv0_forVis['%sFace' % meshName],
                )


def main():
    # general
    testDataNickName = os.environ['DS']
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    approachEntryDict = getApproachEntryDict()
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)
    testDataEntry = testDataEntryDict[testDataNickName]

    # from testDataEntry
    datasetObj = testDataEntry['datasetObj']
    dataset = datasetObj.datasetConf['dataset']
    indChosen = testDataEntry['indChosen']
    indVisChosen = testDataEntry['indVisChosen']
    flagSplit = datasetObj.flagSplit

    # html setups
    visualDir = projRoot + 'cache/B_%s/%s_dumpHtmlForPrepick/' % (Btag, testDataNickName)
    mkdir_full(visualDir)
    htmlStepper = HTMLStepper(visualDir, 100, testDataNickName)

    for j in indVisChosen:
        print('Processing index %d' % j)
        bsv0_forVis_dict = {}
        for approachNickName in approachEntryDict.keys():
            approachEntry = approachEntryDict[approachNickName]
            approachShownName = approachEntry['approachShownName']
            scriptTag = approachEntry['scriptTag']

            outputVis_root = Btag_root + 'vis/%s/%s_%s/' % \
                             (scriptTag, testDataNickName, approachNickName)
            outputVisFileName = outputVis_root + '%08d.pkl' % j

            if not os.path.isfile(outputVisFileName):
                continue
                # print('File does not exist: %s.' % outputVisFileName)
                # import ipdb
                # ipdb.set_trace()
                # raise ValueError('You cannot proceed.')

            with open(outputVisFileName, 'rb') as f:
                bsv0_forVis = pickle.load(f)
            bsv0_forVis_dict[approachNickName] = bsv0_forVis

        summary0 = OrderedDict([])
        txt0 = []
        brInds = [0, ]
        addToSummary0Txt0BrInds(
            summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict,
            visualDir=visualDir, testDataNickName=testDataNickName, dataset=dataset,
        )
        headerMessage = 'testDataNickName: %s, Dataset: %s, Index: %s, flagSplit: %d' % \
                        (testDataNickName, dataset, bsv0_forVis['index'], bsv0_forVis['flagSplit'])
        subMessage = 'HouseID: %d' % (bsv0_forVis['houseID'])
        htmlStepper.step2(
            summary0, txt0, brInds,
            headerMessage=headerMessage,
            subMessage=subMessage,
        )


if __name__ == '__main__':
    main()
