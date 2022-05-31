# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
import numpy as np
from datasets_registration import datasetDict
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, getRotationMatrixBatchNP
from skimage.io import imread
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert, combineMultiShapes_withVertRgb, \
    addPointCloudToMesh
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP, vertInfo2faceVertInfoTHGPU
from codes_py.toolbox_3D.self_sampling_v1 import mesh_weighted_sampling_given_normal
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.benchmarking_v1 import packageCDF1, packageDepth, affinePolyfitWithNaN
from codes_py.toolbox_3D.draw_mesh_v1 import pickInitialObservationPoints, getUPick0List
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow
import cv2
import os


bt = lambda s: s[0].upper() + s[1:]


class DemoRenderDataset(object):
    def __init__(self, datasetConf, **kwargs):
        # basics
        self.meta = {}

        # kwargs
        self.projRoot = kwargs['projRoot']
        self.datasetSplit = kwargs['datasetSplit']

        # datasetConf
        self.datasetConf = datasetConf

        # layout variables
        dataset = datasetConf['dataset']
        assert dataset.startswith('demo')
        self.dataset = dataset

        # preset
        pass

        # udl
        A1 = udl('pkl_A1_', dataset)
        m = A1['m']
        self.flagSplit = 3 * np.ones((m, ), dtype=np.int32)
        self.indTrain = np.where(self.flagSplit == 1)[0]
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.mTrain = int(len(self.indTrain))
        self.mVal = int(len(self.indVal))
        self.mTest = int(len(self.indTest))

        # must be defined
        for k in ['focalLengthWidth', 'focalLengthHeight', 'winWidth', 'winHeight']:
            # 'fScaleWidth', 'fScaleHeight']:
            # These intrinsics settings should be fixed for the whole training process.
            assert k in datasetConf

    def __len__(self):
        if self.datasetSplit == 'train':
            return self.mTrain
        elif self.datasetSplit == 'val':
            return self.mVal
        elif self.datasetSplit == 'test':
            return self.mTest
        else:
            raise NotImplementedError('Unknown self.datasetSplit: %s' % self.datasetSplit)

    def getOneNP(self, index):
        # layout variables
        datasetConf = self.datasetConf
        did = datasetConf['did']
        dataset = datasetConf['dataset']
        datasetID = datasetDict[dataset]

        b0np = dict(
            index=np.array(index, dtype=np.float32),
            did=np.array(did, dtype=np.float32),
            datasetID=np.array(datasetID, dtype=np.float32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
        )

        winWidth, winHeight = datasetConf['winWidth'], datasetConf['winHeight']
        img0 = udl('png_A1_', dataset, index)
        if img0.shape[-1] == 4:
            img0 = img0[..., :3]
        if not (img0.shape == (winHeight, winWidth, 3)):
            height = img0.shape[0]
            width = img0.shape[1]
            r = min(height / float(winHeight) / 2., width / float(winWidth) / 2.)
            rx = r * winWidth
            ry = r * winHeight
            cxcyrxry0 = np.array([img0.shape[1] / 2., img0.shape[0] / 2., rx, ry], dtype=np.float32)
            b0 = croppingCxcyrxry0(img0, cxcyrxry0, padConst=0)
            c0 = cv2.resize(b0, (winWidth, winHeight), interpolation=cv2.INTER_CUBIC)
        else:
            c0 = img0.copy()
        b0np['imgForUse'] = c0.transpose((2, 0, 1))

        return b0np

    def __getitem__(self, indexTrainValTest):
        if self.datasetSplit == 'train':
            index = self.indTrain[indexTrainValTest]
            firstIndex = self.indTrain[0]
        elif self.datasetSplit == 'val':
            index = self.indVal[indexTrainValTest]
            firstIndex = self.indVal[0]
        elif self.datasetSplit == 'test':
            index = self.indTest[indexTrainValTest]
            firstIndex = self.indTest[0]
        else:
            print('Warning: Since your datasetSplit is %s, you cannot call this function!' %
                  self.datasetSplit)
            raise ValueError

        if self.datasetConf.get('singleSampleOverfittingMode', False):
            index = int(firstIndex)
        else:
            index = int(index)

        b0np = self.getOneNP(index)
        return b0np


class FreeDemoRenderDataset(object):
    def __init__(self, datasetConf, **kwargs):
        # basics
        self.meta = {}

        # kwargs
        self.projRoot = kwargs['projRoot']

        # datasetConf
        self.datasetConf = datasetConf

        # layout variables
        dataset = datasetConf['dataset']
        assert dataset.startswith('freedemo') or dataset.startswith('squareFreedemo')
        self.dataset = dataset

        # preset
        pass

        # os.listdir
        self.imgRoot = self.projRoot + 'v/A/%s/' % dataset
        self.nameList = sorted(os.listdir(self.imgRoot))
        self.flagSplit = 3 * np.ones((len(self.nameList), ), dtype=np.int32)
        assert len(self.nameList) > 0  # Make sure it is not empty

        # must be defined
        for k in ['focalLengthWidth', 'focalLengthHeight', 'winWidth', 'winHeight']:
            # 'fScaleWidth', 'fScaleHeight']:
            # These intrinsics settings should be fixed for the whole training process.
            assert k in datasetConf

    def __len__(self):
        return len(self.nameList)

    def getOneNP(self, index):
        # layout variables
        datasetConf = self.datasetConf
        dataset = datasetConf['dataset']

        b0np = dict(
            index=np.array(index, dtype=np.float32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
            dataset=self.dataset,
        )

        winWidth, winHeight = datasetConf['winWidth'], datasetConf['winHeight']

        b0np['focalLengthWidth'] = np.array(float(datasetConf['focalLengthWidth']), dtype=np.float32)
        b0np['focalLengthHeight'] = np.array(float(datasetConf['focalLengthHeight']), dtype=np.float32)
        b0np['winWidth'] = np.array(float(datasetConf['winWidth']), dtype=np.float32)
        b0np['winHeight'] = np.array(float(datasetConf['winHeight']), dtype=np.float32)
        b0np['fScaleWidth'] = np.array(float(datasetConf['fScaleWidth']), dtype=np.float32)
        b0np['fScaleHeight'] = np.array(float(datasetConf['fScaleHeight']), dtype=np.float32)

        b0np['EWorld'] = np.array([0, 0, 0], dtype=np.float32)
        b0np['LWorld'] = np.array([0, 0, 1], dtype=np.float32)
        b0np['UWorld'] = np.array([0, -1, 0], dtype=np.float32)

        # img0 = udl('png_A1_', dataset, index)
        img0 = imread(self.imgRoot + self.nameList[index])
        if img0.shape[-1] == 4:
            img0 = img0[..., :3]
        assert len(img0.shape) == 3 and img0.shape[2] == 3
        assert img0.max() > 20
        img0 = img0.astype(np.float32) / 255.
        if not (img0.shape == (winHeight, winWidth, 3)):
            height = img0.shape[0]
            width = img0.shape[1]
            r = min(height / float(winHeight) / 2., width / float(winWidth) / 2.)
            rx = r * winWidth
            ry = r * winHeight
            cxcyrxry0 = np.array([img0.shape[1] / 2., img0.shape[0] / 2., rx, ry], dtype=np.float32)
            b0 = croppingCxcyrxry0(img0, cxcyrxry0, padConst=0)
            c0 = cv2.resize(b0, (winWidth, winHeight), interpolation=cv2.INTER_CUBIC)
        else:
            c0 = img0.copy()
        b0np['imgForUse'] = c0.transpose((2, 0, 1))

        return b0np
