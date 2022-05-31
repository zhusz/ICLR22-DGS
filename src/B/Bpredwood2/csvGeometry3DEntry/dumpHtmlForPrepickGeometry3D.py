# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

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
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, drawDepthDeltaSign, to_heatmap
from collections import OrderedDict
import pickle
import numpy as np
# debugging
from codes_py.np_ext.mat_io_v1 import pSaveMat


bt = lambda s: s[0].upper() + s[1:]


def addToSummary0Txt0BrInds(
        summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict, **kwargs):
    # Not Functional
    for approachNickName in approachEntryDict.keys():
        approachEntry = approachEntryDict[approachNickName]
        approachShownName = approachEntry['approachShownName']
        bsv0_forVis = bsv0_forVis_dict[approachNickName]

        for meshName in ['evalPredFit', 'evalViewFit']:
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            if '%s%s%sMeshDrawingPackageNumView' % \
                    (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType)) \
                    not in bsv0_forVis.keys():
                continue

            # rgb / depth raw
            if meshName == 'evalPredFit':
                summary0['%s depthRaw' % approachShownName] = to_heatmap(
                    bsv0_forVis.get('depthRaw', np.ones((1, 10, 10), dtype=np.float32))[0, :, :], vmin=0.5, vmax=6, cmap='inferno',
                )
                txt0.append('')
            elif meshName == 'evalViewFit':
                summary0['%s imgRaw' % approachShownName] = \
                    bsv0_forVis.get('imgRaw', np.ones((3, 10, 10), dtype=np.float32)).transpose((1, 2, 0))
                txt0.append('')
            else:
                raise NotImplementedError('Unknown meshName: %s' % meshName)

            # input image / depth delta sign visualization
            if meshName == 'evalViewFit' or 'depthForUse' not in bsv0_forVis.keys():
                summary0['%s Input Img (%s)' % (approachShownName, meshName)] = \
                    bsv0_forVis['imgForUse'].transpose((1, 2, 0))
                txt0.append('')
            elif meshName == 'evalPredFit':
                '''
                summary0['%s delta sign' % approachShownName] = \
                    drawDepthDeltaSign(
                        bsv0_forVis['depthRenderedFromEvalPredFit2'],
                        bsv0_forVis['depthForUse'][0, :, :],
                        deltaThre=0.05,
                    )
                '''
                summary0['%s predDepth2' % approachShownName] = \
                    to_heatmap(bsv0_forVis['predDepth2'], vmin=0.5, vmax=6, cmap='inferno')
                txt0.append('')
            else:
                raise NotImplementedError('Unknown meshName: %s' % meshName)

            # depth
            if meshName == 'evalPredFit':
                summary0['%s depthPredFit2' % approachShownName] = \
                    to_heatmap(bsv0_forVis['depthRenderedFromEvalPredFit2'],
                              vmin=0.5, vmax=6, cmap='inferno')
            elif meshName == 'evalViewFit':
                summary0['%s depthLabel' % approachShownName] = \
                    to_heatmap(bsv0_forVis['depthForUse'][0, :, :],
                              vmin=0.5, vmax=6, cmap='inferno')
            else:
                raise NotImplementedError('Unknown meshName: %s' % meshName)
            if 'finalBenFitR1' in bsv0_forVis.keys() and meshName == 'evalPredFit':
                txt0.append('AbsDiff: %.3f, AbsRel: %.3f, r1: %.3f' % (
                    bsv0_forVis['finalBenFitAbsDiff'],
                    bsv0_forVis['finalBenFitAbsRel'],
                    bsv0_forVis['finalBenFitR1'],
                ))
            elif meshName == 'evalViewFit':
                t = bsv0_forVis['depthForUse'][0, :, :]
                t = t[np.isfinite(t)]
                txt0.append('Label Depth: NaNMin: %.3f, NaNMax: %.3f, NaNMean: %.3f' %
                            (t.min(), t.max(), t.max()))
            else:
                txt0.append('')

            # floor plan
            summary0['%s %s Floor Plan' % (approachShownName, meshName)] = \
                bsv0_forVis['%s%s%sMeshDrawingPackageFloorPlan' %
                            (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))]
            if 'finalBenFitF1' in bsv0_forVis.keys() and meshName == 'evalPredFit':
                txt0.append('CD: %.3f, P: %.3f, R: %.3f, F1: %.3f' %
                            (bsv0_forVis['finalBenFitChamfer'],
                             bsv0_forVis['finalBenFitPrec'],
                             bsv0_forVis['finalBenFitRecall'],
                             bsv0_forVis['finalBenFitF1']))
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
    return summary0, txt0, brInds


def dump(bsv0_forVis_dict, **kwargs):
    dumpDir = kwargs['dumpDir']

    # total
    bsv0_forVis = list(bsv0_forVis_dict.values())[0]
    for meshName in ['evalView']:
        sysLabel = 'cam'
        if ('%sVert%s' % (meshName, bt(sysLabel))) in bsv0_forVis.keys():
            tmp = np.clip(bsv0_forVis['%sVertRgb' % meshName], a_min=0, a_max=1)
            tmp[np.isfinite(tmp) == 0] = 0
            print('L109 %.3f %.3f' % (tmp.min(), tmp.max()))
            dumpPly(
                dumpDir + '%s_%s_%d(%d)_%s_%s.ply' %
                ('label', bsv0_forVis['dataset'],
                 bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                 meshName, sysLabel),
                bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                bsv0_forVis['%sFace' % meshName],
                tmp,
            )

    # pre approach
    for approachNickName in bsv0_forVis_dict.keys():
        bsv0_forVis = bsv0_forVis_dict[approachNickName]
        for meshName in ['evalPredMetric', 'evalViewMetric', 'evalPredFit', 'evalViewFit']:
            sysLabel = 'cam'
            if ('%sVert%s' % (meshName, bt(sysLabel)) in bsv0_forVis.keys()):
                dumpPly2(
                    dumpDir + '%s_%s_%d(%d)_%s_%s.ply' %
                    (approachNickName, bsv0_forVis['dataset'],
                     bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                     meshName, sysLabel),
                    bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                    bsv0_forVis['%sFace' % meshName],
                    bsv0_forVis['%sFaceRgb' % meshName],
                )
            '''
            if ('%sVertRgb' % meshName) not in bsv0_forVis.keys():
                bsv0_forVis['%sVertRgb' % meshName] = \
                    0.6 * np.ones_like(bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))])
            tmp = np.clip(bsv0_forVis['%sVertRgb' % meshName], a_min=0, a_max=1)
            tmp[np.isfinite(tmp) == 0] = 0
            print('L129 %.3f %.3f' % (tmp.min(), tmp.max()))
            dumpPly(
                dumpDir + '%s_%s_%d(%d)_%s_%s.ply' %
                (approachNickName, bsv0_forVis['dataset'],
                 bsv0_forVis['index'], bsv0_forVis['flagSplit'],
                 meshName, sysLabel),
                bsv0_forVis['%sVert%s' % (meshName, bt(sysLabel))],
                bsv0_forVis['%sFace' % meshName],
                tmp,
            )
            '''


def main():
    # set your set - testDataEntryKey
    testDataNickName = os.environ['DS']
    # testDataNickName = 'scannetOfficialTestSplit10'
    # testDataNickName = 'rebuttalTimeMpv1rebuttal'
    # testDataNickName = 'mpv1'
    # testDataNickName = 'demo1'
    # testDataNickName = 'freedemo1'
    # testDataNickName = 'freedemo2'
    # testDataNickName = 'freedemo3'
    # testDataNickName = 'pix3d'

    # general
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
    dumpDir = visualDir
    os.makedirs(dumpDir, exist_ok=True)

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
                print('File does not exist: %s.' % outputVisFileName)
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')

            with open(outputVisFileName, 'rb') as f:
                bsv0_forVis = pickle.load(f)
            bsv0_forVis_dict[approachNickName] = bsv0_forVis

        summary0 = OrderedDict([])
        txt0 = []
        brInds = [0, ]
        summary0, txt0, brInds = addToSummary0Txt0BrInds(
            summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict,
            visualDir=visualDir, testDataNickName=testDataNickName, dataset=dataset,
        )
        headerMessage = 'testDataNickName: %s, Dataset: %s, Index: %s, flagSplit: %d' % \
                        (testDataNickName, dataset, bsv0_forVis['index'], bsv0_forVis['flagSplit'])
        subMessage = 'HouseID: %d' % (bsv0_forVis.get('houseID', -1))
        htmlStepper.step2(
            summary0, txt0, brInds,
            headerMessage=headerMessage,
            subMessage=subMessage,
        )
        dump(bsv0_forVis_dict, dumpDir=dumpDir)


if __name__ == '__main__':
    main()
