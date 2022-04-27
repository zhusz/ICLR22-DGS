# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

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
from PIL import Image
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'external_codes/omnidata_tools/omnidata_tools/torch/')

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform


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
    # extMethodologyName = 'extOmnidatatoolsdemoPretrainedDptHybrid'
    # extMethodologyName = 'extOmnidatatoolsdemoPretrainedDptLarge'
    # extMethodologyName = 'extOmnidatatoolsdemoDptHybridTrain12261542'
    # extMethodologyName = 'extOmnidatatoolsdemoDptHybridTrain12271705'
    extMethodologyName = 'extOmnidatatoolsdemoDptHybridTrain12281111'
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
    if extMethodologyName == 'extOmnidatatoolsdemoPretrainedDptHybrid':
        pretrained_weights_path = projRoot + \
            'external_codes/omnidata_tools/omnidata_tools/torch/pretrained_models/' + \
            'omnidata_rgb2depth_dpt_hybrid.pth'
        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    elif extMethodologyName == 'extOmnidatatoolsdemoPretrainedDptLarge':
        pretrained_weights_path = projRoot + \
                                  'external_codes/omnidata_tools/omnidata_tools/torch/pretrained_models/' + \
                                  'omnidata_rgb2depth_dpt_large.pth'
        model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    elif extMethodologyName == 'extOmnidatatoolsdemoDptHybridTrain12261542':
        pretrained_weights_path = projRoot + \
                                  'external_codes/omnidata_tools/omnidata_tools/torch/trained/' + \
                                  'train_12261542.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    elif extMethodologyName == 'extOmnidatatoolsdemoDptHybridTrain12271705':
        pretrained_weights_path = projRoot + \
                                  'external_codes/omnidata_tools/omnidata_tools/torch/trained/' + \
                                  'train_12271705.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    elif extMethodologyName == 'extOmnidatatoolsdemoDptHybridTrain12281111':
        pretrained_weights_path = projRoot + \
                                  'external_codes/omnidata_tools/omnidata_tools/torch/trained/' + \
                                  'train_12281111.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    else:
        raise NotImplementedError('Unknown extMethodologyName: %s' % extMethodologyName)
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size),  # , interpolation=PIL.Image.BILINEAR),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=0.5, std=0.5)])

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
            img0 = bsv0_initial['imgForUse'].transpose((1, 2, 0))
            output0 = test_outputs(img0, model, trans_totensor, cudaDevice=cudaDevice)
            output0 = output0 * 10  # inherient model bias
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
                summary0['img'] = img0
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


def test_outputs(img0, model, trans_totensor, cudaDevice='cuda:0'):
    # assume img0 is between 0 (0) and 1 (255), and is float32 dtype.
    epsilon = 0.15  # 1. / 127.  # mainly rule out [0, 255] or [-1, 1]
    assert img0.min() >= -epsilon, img0.min()
    assert img0.max() <= 1 + epsilon, img0.max()
    assert len(img0.shape) == 3 and img0.shape[2] == 3

    # img_tensor = torch.from_numpy(img0.transpose((
    #     2, 0, 1)) * 2. - 1.)[None, :, :, :].float().to(cudaDevice)
    # img_tensor = trans_totensor(img_tensor)
    pil0 = Image.fromarray((img0 * 255.).astype(np.uint8))
    img_tensor = trans_totensor(pil0)[:3].unsqueeze(0).to(cudaDevice)
    # img_tensor = trans_totensor(img0.transpose((2, 0, 1)))[None, :, :, :].to(cudaDevice)

    output = model(img_tensor).clamp(min=0, max=1)
    # output = 1. / (output + 1.e-6)
    # output_vis = depth_to_heatmap(output[0, :, :].detach().cpu().numpy())
    # output_vis = (output_vis - output_vis.min()) / (output_vis.max() - output_vis.min())
    return output[0, :, :].detach().cpu().numpy()


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


