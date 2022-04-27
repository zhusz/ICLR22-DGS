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
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
import numpy as np
import PIL.Image
import io
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from collections import OrderedDict
from easydict import EasyDict
from codes_py.template_visualStatic.visualStatic_v1 import TemplateVisualStatic
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
import pickle


class Visual(TemplateVisualStatic):
    pass


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    corenetRawRoot = projRoot + 'remote_fastdata/corenet/data/'
    visualDir = projRoot + 'cache/dataset/%s/A1b/' % dataset
    mkdir_full(visualDir)
    htmlStepper = HTMLStepperNoPrinting(visualDir, 50, 'A1b')
    cudaDevice = 'cuda:0'

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    flagSplit = A1['flagSplit']
    sptTagList = A1['sptTagList']
    splitTagList = A1['splitTagList']
    corenetShaH3TagList = A1['corenetShaH3TagList']
    corenetShaTagList = A1['corenetShaTagList']
    camObj2World = A1['camObj2World']
    camWorld2Cam = A1['camWorld2Cam']
    cameraTransform = A1['cameraTransform']
    catIDList = A1['catIDList']
    shaIDList = A1['shaIDList']
    del A1

    # cameraTransform are actually the same for all the samples
    for i in range(4):
        for j in range(4):
            tmp = np.unique(cameraTransform[:, i, j])
            assert len(tmp) == 1
    f0 = float(cameraTransform[0, 0, 0])

    # all the camWorld2Cam(Theirs) are det 1 (which means they are rotation matrices)
    # det = np.linalg.det(camWorld2Cam)
    # This does not hold - many camWorld2Cam0 have non-one determinant.
    # So this means, you should not record any world sys coordinates (they are scale variant)

    # Now do the real business

    # 1. Inverse mapping from corenetShaTag to index
    corenetShaTagDict = {}
    for j, corenetShaTag in enumerate(corenetShaTagList):
        corenetShaTagDict[corenetShaTag] = j

    # 2. One step from obj sys to camOurs sys
    camObj2CamTheirs = np.matmul(camWorld2Cam, camObj2World)
    camCamTheirs2CamOurs0 = np.eye(4).astype(np.float32)
    camCamTheirs2CamOurs0[0, 3] = -0.5
    camCamTheirs2CamOurs0[1, 1] = -1.
    camCamTheirs2CamOurs0[1, 3] = +0.5
    camCamTheirs2CamOurs0[2, 3] = f0 / 2.
    camCamTheirs2CamOurs = np.tile(camCamTheirs2CamOurs0, (m, 1, 1))
    camObj2CamOurs = np.matmul(camCamTheirs2CamOurs, camObj2CamTheirs)
    assert np.all(camObj2CamOurs[:, 3, :3] == 0)
    assert np.all(camObj2CamOurs[:, 3, 3] == 1)
    camRObj2CamOurs = camObj2CamOurs[:, :3, :3]
    camTObj2CamOurs = camObj2CamOurs[:, :3, 3]

    # 3. Complementing useful supplementary data
    winWidth = 256 * np.ones((m, ), dtype=np.int32)
    winHeight = 256 * np.ones((m, ), dtype=np.int32)
    focalLengthWidth = 128 * f0 * np.ones((m, ), dtype=np.float32)
    focalLengthHeight = 128 * f0 * np.ones((m, ), dtype=np.float32)

    # render the indChosen
    indChosen = []  # list(range(5))
    for j in indChosen:
        print('A1b %s indChosen %d' % (dataset, j))

        sptTag = sptTagList[j]
        splitTag = splitTagList[j]
        corenetShaH3Tag = corenetShaH3TagList[j]
        corenetShaTag = corenetShaTagList[j]
        catID = catIDList[j]
        shaID = shaIDList[j]
        camRObj2CamOurs0 = camRObj2CamOurs[j]
        camTObj2CamOurs0 = camTObj2CamOurs[j]
        winWidth0 = winWidth[j]
        winHeight0 = winHeight[j]
        focalLengthWidth0 = focalLengthWidth[j]
        focalLengthHeight0 = focalLengthHeight[j]
        npz = dict(np.load(
            projRoot + 'remote_fastdata/corenet/data/%s.%s/%s/%s.npz' %
                       (sptTag, splitTag, corenetShaH3Tag, corenetShaTag)))
        img0 = np.array(PIL.Image.open(io.BytesIO(npz['pbrt_image'])), dtype=np.float32) / 255.

        vertObj0, face0 = load_obj_np(
            projRoot + 'remote_fastdata/shapeNetV2/ShapeNetCore.v2/%s/%s/models/model_normalized.obj' %
            (catID, shaID),
        )
        vertObj0 = vertObj0.astype(np.float32)
        vertCamOurs0 = np.matmul(vertObj0, camRObj2CamOurs0.transpose()) + camTObj2CamOurs0[None, :]

        bsv0 = {}
        bsv0['progressedVertCam'] = vertCamOurs0
        bsv0['progressedFace'] = face0
        bsv0['ECam'] = np.array([0, 0, 0], dtype=np.float32)
        bsv0['LCam'] = np.array([0, 0, 1], dtype=np.float32)
        bsv0['UCam'] = np.array([0, -1, 0], dtype=np.float32)
        bsv0['amodalDepthMax'] = 4
        bsv0['winHeight'] = winHeight0
        bsv0['winWidth'] = winWidth0
        bsv0['focalLengthHeight'] = focalLengthHeight0
        bsv0['focalLengthWidth'] = focalLengthWidth0

        for k in bsv0.keys():
            if type(bsv0[k]) == np.ndarray:
                print('%s %s' % (k, bsv0[k].dtype))

        bsv0 = Visual.stepDrawMeshPackage(
            bsv0, inputMeshName='progressed', outputMeshDrawingName='progressed',
            usingObsFromMeshDrawingName=None, numView=3, meshVertInfoFaceInfoType='plain',
            sysLabel='cam', cudaDevice=cudaDevice, verboseGeneral=1)

        summary = OrderedDict([])
        txt0 = []
        brInds = [0]
        summary['img0'] = img0[None, :, :, :]
        txt0.append('')
        for v in range(bsv0['progressedCamPlainDrawing0'].shape[0]):
            summary['progressedView%d' % v] = bsv0['progressedCamPlainDrawing0'][v:v + 1, :, :, :]
            txt0.append('')
        visual = EasyDict()
        visual.m = 1
        visual.summary = summary
        visual.index = [j]
        visual.insideBatchIndex = [0]
        visual.dataset = [dataset]
        visual.datasetID = [-1]
        htmlStepper.step(-1, None, visual, [txt0], brInds)

        dumpPly(
            visualDir + 'progressedCamMesh_%d.ply' % j,
            vertCamOurs0, face0,
        )

    with open(projRoot + 'v/A/%s/A1b_order1Supp.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'flagSplit': flagSplit,
            'sptTagList': sptTagList,
            'splitTagList': splitTagList,
            'corenetShaH3TagList': corenetShaH3TagList,
            'corenetShaTagList': corenetShaTagList,
            'corenetShaTagDict': corenetShaTagDict,

            'camObj2World': camObj2World,
            'camWorld2CamTheirs': camWorld2Cam,  # This cam actually means camTheirs
            'cameraTransform': cameraTransform,
            'camRObj2CamOurs': camRObj2CamOurs,
            'camTObj2CamOurs': camTObj2CamOurs,

            'winWidth': winWidth,
            'winHeight': winHeight,
            'focalLengthWidth': focalLengthWidth,
            'focalLengthHeight': focalLengthHeight,
            'f0': f0,

            'catIDList': catIDList,
            'shaIDList': shaIDList,
        }, f)


if __name__ == '__main__':
    main()
