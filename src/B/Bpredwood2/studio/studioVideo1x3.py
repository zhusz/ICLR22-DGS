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
from codes_py.toolbox_framework.framework_util_v4 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import tabularPrintingConstructing, tabularCvsDumping
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, drawDepthDeltaSign
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, getRotationMatrixBatchNP, \
    ELU2cam
from codes_py.toolbox_show_draw.draw_v1 import drawBoxXDXD
from collections import OrderedDict
from skimage.io import imsave
from skvideo.io import vwrite
import cv2
import time
import pickle
import numpy as np
np.set_printoptions(suppress=True)


bt = lambda s: s[0].upper() + s[1:]


def main():
    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=None)
    approachEntryDict = getApproachEntryDict()
    Btag = os.path.dirname(os.path.realpath(__file__)).split('/')[-2]
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    # output
    studioTag = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    studioSubTag = os.path.basename(os.path.realpath(__file__))[:-3]
    assert os.path.isdir(projRoot + 'cache/')
    visualDir = projRoot + 'cache/B_%s/%s/%s/' % (Btag, studioTag, studioSubTag)
    os.makedirs(visualDir, exist_ok=True)

    # ---------------------- studio specific scripts below --------------------- #

    # 221: input image, 222: occnet+georeg, 223: disn+tsdfvox, 224: ours
    taskPool = [  # (testDataNickName, index (of the dataset), c_distance, l_distance)
        # ('squareFreedemo1', 0, 2.25, 4),
        # ('squareFreedemo1', 1, 2.25, 4),
        # ('squareFreedemo1', 2, 2.3, 4),
        # ('squareFreedemo1', 3, 2.3, 4),
        # ('squareFreedemo1', 7, 2.25, 4),
        # ('squareFreedemo2', 17, 2.1, 4), ('squareFreedemo2', 22, 1.7, 3),
        # ('squareFreedemo3', 16, 1.5, 2.9)

        ('squareFreedemo1', 0, 2.25, 4),
        ('squareFreedemo1', 7, 2.25, 4),
        ('squareFreedemo2', 15, 1.3, 2.4),
        ('squareFreedemo2', 16, 1.2, 2.2),
        ('squareFreedemo2', 17, 1.3, 2.4),
        ('squareFreedemo2', 22, 1.6, 3),
        ('squareFreedemo3', 5, 2, 3.7),
        ('squareFreedemo3', 16, 1.5, 2.9),
        ('squareFreedemo3', 35, 2.2, 4),
    ]
    approachNickName234 = []
    # colorPool = [[0.7, 0.7, 0.3], [0.7, 0.3, 0.3], [0.3, 0.3, 0.7]]
    colorPool = [[0.3, 0.3, 0.7], [0.3, 0.2, 0.8]]
    colorPool = [np.array(c, dtype=np.float32) for c in colorPool]
    txtPool = ['Baseline DGS', 'Ours ASC']
    omega = 60  # 60 degrees per second
    frameRate = 25  # 25 frames per second
    degreesPerFrame = float(omega) / frameRate
    videoDuration = 6  # 10 seconds
    nFrame = int(videoDuration * frameRate)
    degreesTotal = degreesPerFrame * nFrame

    scaling = 2
    fxywxy = np.array([230.4, 230.4, 256, 256], dtype=np.float32) * scaling
    height = 256 * scaling
    width = 256 * scaling
    intervalWidth = 10 * scaling
    intervalHeight = 25 * scaling
    C = 3
    R = 1
    cloth = np.ones(
        (nFrame,
         R * (height + intervalHeight),  # - intervalHeight,
         C * (width + intervalWidth) - intervalWidth, 3),
        dtype=np.float32)
    debug1 = np.ones(
        (nFrame,
         R * (height + intervalHeight) - intervalHeight,
         C * (width + intervalWidth) - intervalWidth, 3),
        dtype=np.float32)
    pyrenderManager = PyrenderManager(width, height)

    t1 = int(os.environ['T1'])
    t2 = int(os.environ['T2'])

    # for testDataNickName, index in taskPool:
    for taskID in range(t1, t2):
        testDataNickName = taskPool[taskID][0]
        index = taskPool[taskID][1]
        c_distance = taskPool[taskID][2]
        l_distance = taskPool[taskID][3]

        input_image = None  # to be set
        for subplotCount in [2, 3]:
            approachNickName = approachNickName234[subplotCount - 2]
            color0 = colorPool[subplotCount - 2]
            txt0 = txtPool[subplotCount - 2]
            r = (subplotCount - 1) // 3
            c = (subplotCount - 1) % 3

            scriptTag = approachEntryDict[approachNickName]['scriptTag']

            outputVis_root = Btag_root + 'vis/%s/%s_%s/' % \
                             (scriptTag, testDataNickName, approachNickName)
            outputVisFileName = outputVis_root + '%08d.pkl' % index
            if not os.path.isfile(outputVisFileName):
                print('File does not exist: %s.' % outputVisFileName)
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')
            with open(outputVisFileName, 'rb') as f:
                bsv0_forVis = pickle.load(f)

            if input_image is None:
                input_image = bsv0_forVis['imgForUse'].transpose((1, 2, 0))
            else:
                pass
                # assert np.all(input_image == bsv0_forVis['imgForUse'].transpose((1, 2, 0)))

            vertWorld0 = bsv0_forVis['evalPredMetricVertWorld']
            face0 = bsv0_forVis['evalPredMetricFace']

            EWorld0 = bsv0_forVis['EWorld']
            LWorld0 = bsv0_forVis['LWorld']
            UWorld0 = bsv0_forVis['UWorld']
            EL0 = LWorld0 - EWorld0
            ELunit0 = EL0 / float(np.linalg.norm(EL0, ord=2, axis=0))
            CWorld0 = EWorld0 + c_distance * ELunit0
            LWorld0 = EWorld0 + l_distance * ELunit0

            alpha_t = np.linspace(0, degreesTotal - degreesPerFrame, nFrame).astype(np.float32)
            CE0 = EWorld0 - CWorld0
            CE_t = np.tile(CE0[None, :], (nFrame, 1))
            rotMat_t = getRotationMatrixBatchNP(
                np.tile(UWorld0[None, :], (nFrame, 1)),
                alpha_t,
            )
            CEprime_t = (rotMat_t * CE_t[:, None, :]).sum(2)  # + 0.4 * UWorld0[None, :]
            Eprime_t = CWorld0[None, :] + CEprime_t
            videoCam_t = ELU2cam(np.concatenate([
                Eprime_t,
                np.tile(LWorld0[None, :], (nFrame, 1)),
                np.tile(UWorld0[None, :], (nFrame, 1)),
            ], 1))
            videoCamInv_t = np.linalg.inv(videoCam_t)

            for t in range(nFrame):
                if t % 10 == 0:
                    print('Processing %s %s %s %d: subplotCount %d, t %d / total %d. ' %
                          (studioTag, studioSubTag, testDataNickName, index,
                           subplotCount, t, nFrame))

                if subplotCount == 2:  # The first running 
                    assert input_image is not None
                    if t == 0:
                        input_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_CUBIC)
                    cloth[t, :height, :width, :] = input_image[None, :, :, :]
                    cv2.putText(
                        cloth[t, :, :, :], 'Input RGB',
                        (int(width * 0.3), int(height + intervalHeight - height * 0.05))                       ,
                        cv2.FONT_HERSHEY_TRIPLEX, height / 500., [0, 0, 0]
                    )
                
                pyrenderManager.clear()
                pyrenderManager.add_point_light(
                    pointLoc=1.2 * EWorld0 - 0.2 * LWorld0, intensity=0.2,
                    color=(color0 * 255.).tolist(),
                )
                pyrenderManager.add_vertRgb_mesh_via_faceRgb(
                    vertWorld0, face0,
                    0.8 * np.ones_like(face0).astype(np.float32),
                )
                pyrenderManager.add_camera(
                    fxywxy, videoCamInv_t[t, :, :],
                )
                tmp0 = pyrenderManager.render()[0].astype(np.float32) / 255.

                cloth[
                    t,
                    r * (height + intervalHeight):r * (height + intervalHeight) + height,
                    c * (width + intervalWidth):c * (width + intervalWidth) + width,
                    :
                ] = tmp0
                cv2.putText(
                    cloth[t, :, :, :], txt0,
                    (int(c * (width + intervalWidth) + width * 0.3), int((r + 1) * (height + intervalHeight) - height * 0.05)),
                    cv2.FONT_HERSHEY_TRIPLEX, height / 500., [0, 0, 0]
                )

                # debug1
                def f(ax):
                    ax.scatter(vertWorld0[:, 0], vertWorld0[:, 2], s=0.02, c='r', marker='.')
                    ax.scatter(Eprime_t[t, 0], Eprime_t[t, 2], s=20, c='m', marker='o')
                    ax.scatter(CWorld0[0], CWorld0[2], s=20, c='b', marker='x')
                    ax.scatter(LWorld0[0], LWorld0[2], s=20, c='k', marker='o')
                    ax.set_xlim(-3, 3)
                    ax.set_ylim(0, 6)
                tmp0 = getPltDraw(f)
                tmp0 = cv2.resize(tmp0, (width, height), interpolation=cv2.INTER_CUBIC)
                debug1[
                    t,
                    r * (height + intervalHeight):r * (height + intervalHeight) + height,
                    c * (width + intervalWidth):c * (width + intervalWidth) + width,
                    :
                ] = tmp0
                if t == 0:
                    imsave(
                        visualDir + '%s_%s_%s_%d_firstFrame.png' %
                            (studioTag, studioSubTag, testDataNickName, index),
                        cloth[0, :, :, :],
                    )
                    if subplotCount == 2:  # the first run
                        imsave(
                            visualDir + '%s_%s_%s_%d_inputImage.png' %
                                (studioTag, studioSubTag, testDataNickName, index),
                            input_image,
                        )

        cloth = np.clip(cloth, a_min=0, a_max=1)
        vwrite(
            visualDir + '%s_%s_%s_%d_cloth.avi' %
                (studioTag, studioSubTag, testDataNickName, index),
            cloth * 255.,
        )

        debug1 = np.clip(debug1, a_min=0, a_max=1)
        vwrite(
            visualDir + '%s_%s_%s_%d_debug1.avi' %
            (studioTag, studioSubTag, testDataNickName, index),
            debug1 * 255.,
        )

        vwrite(
            visualDir + '%s_%s_%s_%d_oursOnly.avi' %
            (studioTag, studioSubTag, testDataNickName, index),
            cloth[:, :height, -width:, :] * 255.,
        )


if __name__ == '__main__':
    main()