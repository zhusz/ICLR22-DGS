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
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, drawDepthDeltaSign
from collections import OrderedDict
from skimage.io import imsave
import pickle
import numpy as np


bt = lambda s: s[0].upper() + s[1:]


def main():
    testDataNickName = os.environ['DS']

    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
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

    # single testDataEntryKey
    # testDataNickName = 'scannetOfficialTestSplit10'
    # testDataNickName = 'demo1'
    # testDataNickName = 'freedemo1'
    testDataEntry = testDataEntryDict[testDataNickName]
    datasetObj = testDataEntry['datasetObj']
    dataset = datasetObj.datasetConf['dataset']

    # single approachEntryKey
    methodologyName = ''
    approachEntry = approachEntryDict[methodologyName]
    scriptTag = approachEntry['scriptTag']

    # cloth
    interval = 5
    w = 256
    h = 256
    R = 3
    C = 7
    cloth = np.ones(((h + interval) * R - interval,
                     (w + interval) * C - interval,
                     3), dtype=np.float32)

    for r, j in enumerate([0, 1, 2]):
        print('Processing Row %d' % r)

        # loading
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

        c = 0
        cloth[
            (h + interval) * r:(h + interval) * r + h,
            (w + interval) * c:(w + interval) * c + w,
            :
        ] = bsv0_forVis['imgForUse'].transpose((1, 2, 0))

        for c in range(1, 7):
            v = c - 1
            metricOrFit = 'fit'
            meshName = 'evalPred'
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            cloth[
                (h + interval) * r:(h + interval) * r + h,
                (w + interval) * c:(w + interval) * c + w,
                :
            ] = bsv0_forVis['%s%s%s%sMeshDrawingPackage%s' %
                            (meshName, bt(metricOrFit), bt(sysLabel), bt(meshVertInfoFaceInfoType),
                             'ViewColor')][v, :, :, :]

    cloth = np.clip(cloth, a_min=0, a_max=1)
    imsave(visualDir + studioTag + '_' + testDataNickName + '.png', cloth)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()