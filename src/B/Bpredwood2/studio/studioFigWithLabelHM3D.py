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
from codes_py.toolbox_show_draw.draw_v1 import drawBoxXDXD
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, drawDepthDeltaSign
from collections import OrderedDict
from skimage.io import imsave
import cv2
import pickle
import numpy as np


bt = lambda s: s[0].upper() + s[1:]


def main():
    testDataNickName = 'hmFsOfficialValSplit10'

    # general
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    approachEntryDict = getApproachEntryDict()
    Btag = os.path.realpath(__file__).split('/')[-3]
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    # output
    studioTag = os.path.basename(os.path.realpath(__file__))[:-3]
    visualDir = projRoot + 'cache/B_%s/%s/' % (Btag, studioTag)
    mkdir_full(visualDir)

    # ---------------------- studio specific scripts below --------------------- #

    # set your set - data
    ind = np.array([0, 1], dtype=np.int32)

    # set your set - method
    methodologyNameList = []

    # set your set - view
    views = [0, 1, 2, 3, 4, 5]

    # single testDataEntryKey
    testDataEntry = testDataEntryDict[testDataNickName]
    datasetObj = testDataEntry['datasetObj']
    dataset = datasetObj.datasetConf['dataset']
    indVisChosen = np.array(testDataEntry['indVisChosen'], dtype=np.int32)[ind]

    # cloth
    intervalW = 70
    intervalH = 25
    w = 256
    h = 256
    pw = int(72 * 1.2)
    ph = int(54 * 1.2)
    R = len(views) * len(ind)
    C = 1 + len(methodologyNameList) + 1  # input, methods, GT
    cloth = np.ones(((h + intervalH) * R,  # difference due to the up-right corner
                     (w + intervalW) * C - intervalW,
                     3), dtype=np.float32)

    for i, j in enumerate(indVisChosen):
        # Input image
        bsv0_forVis = loadResult(
            Btag_root, approachEntryDict[methodologyNameList[0]]['scriptTag'],
            testDataNickName, methodologyNameList[0], j)
        r = i * len(views)
        c = 0
        cloth[
            (h + intervalH) * r + intervalH:(h + intervalH) * r + h + intervalH,  # difference up-right
            (w + intervalW) * c:(w + intervalW) * c + w,
            :
        ] = bsv0_forVis['imgForUse'].transpose((1, 2, 0))

        # each approach
        for ic, methodologyName in enumerate(methodologyNameList):
            bsv0_forVis = loadResult(
                Btag_root, approachEntryDict[methodologyName]['scriptTag'],
                testDataNickName, methodologyName, j
            )
            for iv, v in enumerate(views):
                r = i * len(views) + iv
                c = 1 + ic
                cloth[
                    (h + intervalH) * r + intervalH:(h + intervalH) * r + h + intervalH,  # difference
                    (w + intervalW) * c:(w + intervalW) * c + w,
                    :
                ] = bsv0_forVis['evalPredFitWorldVertRgbMeshDrawingPackageViewColor'][v, :, :, :]

                # up-right corner
                offset = 5
                top = (h + intervalH) * r + offset
                down = (h + intervalH) * r + ph + offset
                left = (w + intervalW) * (c + 1) - pw - offset
                right = (w + intervalW) * (c + 1) - 5
                cloth[
                    top:down,
                    left:right,
                    :
                ] = cv2.resize(
                    bsv0_forVis['evalViewFitWorldVertRgbMeshDrawingPackageViewColor'][v, :, :, :],
                    (pw, ph), interpolation=cv2.INTER_CUBIC
                )
                cloth = drawBoxXDXD(
                    cloth, np.array([left, right, top, down], dtype=np.float32), lineWidthFloat=2.,
                    rgb=np.array([0, 0.5, 1], dtype=np.float32),
                )

        # ground truth
        bsv0_forVis = loadResult(
            Btag_root, approachEntryDict[methodologyNameList[0]]['scriptTag'],
            testDataNickName, methodologyNameList[0], j)
        for iv, v in enumerate(views):
            r = i * len(views) + iv
            c = 1 + len(methodologyNameList)
            cloth[
                (h + intervalH) * r + intervalH:(h + intervalH) * r + h + intervalH,  # difference
                (w + intervalW) * c:(w + intervalW) * c + w,
                :
            ] = bsv0_forVis['evalViewFit2WorldVertRgbMeshDrawingPackageViewColor'][v, :, :, :]

    cloth = np.clip(cloth, a_min=0, a_max=1)
    fn = visualDir + studioTag + '_%s' % testDataNickName
    for i in ind:
        fn += ('_' + '%d' % i)
    fn += '.png'
    imsave(fn, cloth)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def loadResult(Btag_root, scriptTag, testDataNickName, methodologyName, j):
    outputVis_root = Btag_root + 'vis/%s/%s_%s/' % \
                     (scriptTag, testDataNickName, methodologyName)
    outputVisFileName = outputVis_root + '%08d.pkl' % j
    if not os.path.isfile(outputVisFileName):
        print('File does not exist: %s.' % outputVisFileName)
        import ipdb
        ipdb.set_trace()
        raise ValueError('You cannot proceed.')
    with open(outputVisFileName, 'rb') as f:
        bsv0_forVis = pickle.load(f)
    return bsv0_forVis


if __name__ == '__main__':
    main()
