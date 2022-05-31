# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
from codes_py.toolbox_3D.representation_v1 import voxSdf2mesh_skmc
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
import os
import pickle
import numpy as np
import copy
import sys

from codes_py.toolbox_show_draw.draw_v1 import to_heatmap
import pickle
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom, \
    constructInitialBatchStepVis0, mergeFromBatchVis
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from collections import OrderedDict
from easydict import EasyDict
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.py_ext.misc_v1 import mkdir_full
from skimage.io import imread

projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'external_codes/MiDaS/')
import glob
import torch
import utils
import cv2
import argparse
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def testExtDemo(testDataEntry):
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'

    # from testDataEntry
    datasetObj = testDataEntry['datasetObj']
    indChosen = testDataEntry['indChosen']
    indVisChosen = testDataEntry['indVisChosen']
    testDataNickName = testDataEntry['testDataNickName']
    flagSplit = datasetObj.flagSplit
    datasetMeta = testDataEntry['meta']

    # visualization
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    fn_python = os.path.basename(__file__)
    assert fn_python.endswith('.py')
    assert fn_python.startswith('ext')
    testDataNickName = testDataEntry['testDataNickName']
    # extMethodologyName = fn_python[:-3]

    # set your set
    cudaDevice = 'cuda:0'
    extMethodologyName = 'extMidasPretrainedDptLarge'
    # extMethodologyName = 'extMidasPretrainedDptHybrid'  This is not able to run correctly
    assert extMethodologyName.startswith(fn_python[:-3])

    # additional feeds (requested by omnidata test demo)
    device = cudaDevice
    # root_dir = projRoot + 'external_codes/omnidata-tools/' \
    #                       'omnidata_tools/torch/pretrained_models/'
    map_location = (lambda storage, loc: storage.cuda()) \
        if torch.cuda.is_available() else torch.device('cpu')

    outputExtResult_root = Btag_root + 'extResult/%s_%s/' % (
        testDataNickName, extMethodologyName)
    mkdir_full(outputExtResult_root)

    visualDir = projRoot + 'cache/B_%s/%s_%s_ext/' % \
                (Btag, testDataNickName, extMethodologyName)
    mkdir_full(visualDir)
    htmlStepper = HTMLStepper(visualDir, 100, testDataNickName)

    # get target task and model
    if extMethodologyName == 'extMidasPretrainedDptLarge':
        model_path = projRoot + 'external_codes/MiDaS/weights/dpt_large-midas-2f21e586.pt'
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif extMethodologyName == 'extMidasPretrainedDptHybrid':
        model_path = projRoot + 'external_codes/MiDaS/weights/dpt_hybrid-midas-501f0c75.pt'
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise NotImplementedError('Unknown extMethodologyName: %s' % extMethodologyName)
    optimize = True
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()
    if optimize == True:
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
    model.to(device)

    # counter
    j1 = int(os.environ.get('J1', -1))
    j2 = int(os.environ.get('J2', -1))
    if (j1 >= 0) and (j2 >= 0):
        count = j1
    else:
        count = 0

    for i, j in enumerate(indChosen):
        if (j1 >= 0) and (j2 >= 0) and ((i < j1) or (i >= j2)):
            continue

        outputExtResultFileName = outputExtResult_root + '%08d.pkl' % j

        if os.path.exists(outputExtResultFileName):
            print('extResult - Skipping %s (flagSplit %d) - count %d / %d' %
                  (outputExtResultFileName, flagSplit[j], count, len(indChosen)))
            with open(outputExtResultFileName, 'rb') as f:
                bsv0_toStore = pickle.load(f)
        else:
            print('extResult - Working on %s (flagSplit %d) - count %d / %d' %
                  (outputExtResultFileName, flagSplit[j], count, len(indChosen)))
            b0np = datasetObj.getOneNP(j)
            batch_np = {k: b0np[k][None] for k in b0np.keys()}
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis, iterCount=-1, visIndex=0,
                P='PNono', D='DNono', S='SNono', R='RNono',
                verboseGeneral=0,
            )
            bsv0_toStore = copy.deepcopy(bsv0_initial)
            bsv0_initial = mergeFromBatchVis(bsv0_initial, batch_vis)

            imgForUse0 = b0np['imgForUse']
            # compute
            img = imgForUse0.transpose((1, 2, 0))
            img_input = transform({'image': imgForUse0.transpose((1, 2, 0))})['image']
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize == True and device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()
                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                        .squeeze()
                        .cpu()
                        .numpy()
                )
            # metric depth / evaluation
            t1 = 1 / prediction.astype(np.float32)
            output0 = t1

            output0_vis = to_heatmap(output0, cmap='inferno')
            tmp0 = depthMap2mesh(
                output0, datasetObj.datasetConf['fScaleWidth'], datasetObj.datasetConf['fScaleHeight'],
                cutLinkThre=0.02,
            )
            predVertCam0 = tmp0['vertCam0']
            predFace0 = tmp0['face0']
            cam0 = bsv0_initial['cam']
            camR0 = cam0[:3, :3]
            camT0 = cam0[:3, 3]
            predVertWorld0 = np.matmul(predVertCam0 - camT0[None, :], camR0)  # inverse of transpose

            # put into bsv0_toStore
            bsv0_toStore['depthPred'] = output0[None, :, :]
            bsv0_toStore['predVertWorld'] = predVertWorld0
            bsv0_toStore['predVertCam'] = predVertCam0
            bsv0_toStore['predFace'] = predFace0
            with open(outputExtResultFileName, 'wb') as f:
                pickle.dump(bsv0_toStore, f)

            # visualization
            if j in indVisChosen:
                # extApproach specific visualization
                summary0 = OrderedDict([])
                summary0['img'] = bsv0_initial['imgForUse'].transpose(1, 2, 0)
                summary0['depth'] = output0_vis
                htmlStepper.step2(
                    summary0, None, (0, ),
                    headerMessage='extOmnidatatoolsdemo %s-%d(%d)' % (
                        datasetObj.datasetConf['dataset'], j, bsv0_toStore['flagSplit']),
                    subMessage='',
                )
                dumpPly(visualDir + 'predVertCam_%s_%d(%d).ply' % (
                    datasetObj.datasetConf['dataset'], j, bsv0_toStore['flagSplit'],
                ), predVertCam0, predFace0)

        count += 1


def main():
    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    testDataEntryDict = getTestDataEntryDict()

    # set your set - testDataNickName
    testDataNickName = 'scannetOfficialTestSplit10'
    # testDataNickName = 'mpv1rebuttal'
    # testDataNickName = 'demo1'
    # testDataNickName = 'freedemo1'
    # testDataNickName = 'pix3d'
    testDataEntry = testDataEntryDict[testDataNickName]

    testExtDemo(testDataEntry)


if __name__ == '__main__':
    main()




