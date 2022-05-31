# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.representation_v1 import voxSdf2mesh_skmc
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from easydict import EasyDict
import pickle
import numpy as np
import copy

from codes_py.toolbox_show_draw.draw_v1 import to_heatmap
import pickle
from codes_py.toolbox_framework.framework_util_v4 import splitPDSRI, splitPDDrandom, \
    constructInitialBatchStepVis0, mergeFromBatchVis, bsv02bsv
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from collections import OrderedDict
from easydict import EasyDict
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.py_ext.misc_v1 import mkdir_full
from skimage.io import imread

projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'external_codes/AdelaiDepth/LeReS/')
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
from lib.spvcnn_classsification import SPVCNN_CLASSIFICATION
import matplotlib.pyplot as plt
from lib.test_utils import reconstruct_depth
import torchvision.transforms as transforms
import cv2
import os
import argparse
import torch
from lib.test_utils import refine_focal, refine_shift


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose(
            [transforms.ToTensor(),
		     transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model


def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # recover focal
    focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, predicted_focal_2, depth_scale_1


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
    extMethodologyName = 'extAdelaiDepthResNext101'
    # extMethodologyName = 'extAdelaiDepthResNet50'
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
    image_size = 384
    if extMethodologyName == 'extAdelaiDepthResNext101':
        depth_model = RelDepthModel(backbone='resnext101')
        shift_model, focal_model = make_shift_focallength_models()
        depth_model.eval()
        args = EasyDict()
        args.load_ckpt = projRoot + 'external_codes/AdelaiDepth/LeReS/res101.pth'
        assert os.path.isfile(args.load_ckpt)
        # tmp = torhc.load(args.load_ckpt)
        load_ckpt(args, depth_model, shift_model, focal_model)
        depth_model.cuda(cudaDevice)
        shift_model.cuda(cudaDevice)
        focal_model.cuda(cudaDevice)
    else:
        raise NotImplementedError('Unknown extMethodologyName: %s' % extMethodologyName)

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

            batch0_np = datasetObj.getOneNP(j)
            batch_np = bsv02bsv(batch0_np)
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis, iterCount=-1, visIndex=0,
                dataset=None,  # only during training you need to input the dataset here.
                P='PNono', D='DNono', S='SNono', R='RNono',
                verboseGeneral=0,
            )
            bsv0_toStore = copy.deepcopy(bsv0_initial)
            bsv0_initial = mergeFromBatchVis(
                bsv0_initial, batch_vis, dataset=None, visIndex=0)

            # flag1
            imgForUse0 = bsv0_initial['imgForUse']
            rgb = (imgForUse0 * 255.).astype(np.uint8).transpose((1, 2, 0))[:, :, ::-1]
            rgb_c = rgb[:, :, ::-1].copy()
            A_resize = cv2.resize(rgb_c, (448, 448))
            img_torch = scale_torch(A_resize)[None, :, :, :]
            pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
            pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
            assert np.all(np.isfinite(pred_depth_ori))
            # flag2
            shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(rgb, pred_depth_ori,
                                                                           shift_model, focal_model)
            # flag3
            t1 = depth_scaleinv.astype(np.float32).copy()
            fScaleX = datasetObj.datasetConf['fScaleWidth']
            fScaleY = datasetObj.datasetConf['fScaleHeight']
            tmp = depthMap2mesh(t1, fScaleX, fScaleY, cutLinkThre=0.05 * 2)
            vertCam0 = tmp['vertCam0']
            face0 = tmp['face0']
            cam0 = bsv0_initial['cam']
            camR0 = cam0[:3, :3]
            camT0 = cam0[:3, 3]
            vertWorld0 = np.matmul(
                vertCam0 - camT0[None, :], camR0  # transpose of inverse
            )
            output0 = t1

            output0_vis = to_heatmap(output0, cmap='inferno')
            predVertCam0 = vertCam0
            predFace0 = face0
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
    # set your set - testDataNickName
    # testDataNickName = 'scannetOfficialTestSplit10'
    testDataNickName = 'mpv1'
    # testDataNickName = 'mpv1rebuttal'
    # testDataNickName = 'demo1'
    # testDataNickName = 'freedemo1'
    # testDataNickName = 'pix3d'

    # general
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])

    testDataEntry = testDataEntryDict[testDataNickName]

    testExtDemo(testDataEntry)


if __name__ == '__main__':
    main()




