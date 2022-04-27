# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
import numpy as np
import pickle
import PIL.Image
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
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, to_heatmap
from skimage.io import imsave
import time
import cv2
import os
import io


class CorenetChoySingleRenderingDataset(object):
    # potentially being put into data loader with multiple replication in the CPU memory
    # So make sure no caching things in here
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
        assert dataset in ['corenetChoySingleRendering']
        self.dataset = dataset

        # preset
        pass

        # udl
        A1 = udl('pkl_A1b_', dataset)
        self.m = A1['m']
        self.flagSplit = A1['flagSplit']
        self.sptTagList = A1['sptTagList']
        self.splitTagList = A1['splitTagList']
        self.corenetShaH3TagList = A1['corenetShaH3TagList']
        self.corenetShaTagList = A1['corenetShaTagList']
        self.camObj2World = A1['camObj2World']
        self.camWorld2CamTheirs = A1['camWorld2CamTheirs']
        self.camRObj2CamOurs = A1['camRObj2CamOurs']
        self.camTObj2CamOurs = A1['camTObj2CamOurs']
        self.winWidth = A1['winWidth'].astype(np.float32)
        self.winHeight = A1['winHeight'].astype(np.float32)
        self.focalLengthWidth = A1['focalLengthWidth'].astype(np.float32)
        self.focalLengthHeight = A1['focalLengthHeight'].astype(np.float32)
        # self.f0 = A1['f0']
        self.catIDList = A1['catIDList']
        self.shaIDList = A1['shaIDList']
        self.catNameList = A1['catNameList']
        del A1

        # expanding flagSplit
        self.mTrain = (self.flagSplit == 1).sum()
        self.mVal = (self.flagSplit == 2).sum()
        self.mTest = (self.flagSplit == 3).sum()
        self.indTrain = np.where(self.flagSplit == 1)[0]
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.indTrainval = np.concatenate([self.indTrain, self.indVal], 0)

        # meta
        self._getView2voxel()

    def _getView2voxel(self):
        self.meta['view2voxel_thcpu'] = torch.from_numpy(
            np.array([
                [128, 0, 0, 0],
                [0, 128, 0, 0],
                [0, 0, 128, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        )

    def __len__(self):
        if self.datasetSplit == 'train':
            return self.mTrain
        elif self.datasetSplit == 'val':
            return self.mVal
        elif self.datasetSplit == 'test':
            return self.mTest
        elif self.datasetSplit == 'trainval':
            return self.mTrain + self.mVal
        else:
            raise NotImplementedError('Unknown self.datasetSplit: %s' % self.datasetSplit)

    def getOneNP(self, index):
        datasetConf = self.datasetConf

        b0np = dict(
            index=np.array(index, dtype=np.int32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
            dataset=self.dataset,
        )

        b0np = self._getBasic(b0np, index)
        b0np = self._getImg(b0np, index, imageSourceKey=datasetConf.get('imageSourceKey', 'pbrt_image'))

        return b0np

    @staticmethod
    def randomlyDrawSample(numSampling, *args):
        m = args[0].shape[0]
        for x in args:
            assert x.shape[0] == m
        ra = np.random.permutation(m)[:numSampling]
        return tuple([x[ra] for x in args])

    def _getQueryPreSampling(self, batch0_np, index):
        datasetConf = self.datasetConf

        if 'fullsurface' in datasetConf['samplingMethodList']:
            tmp = self.getPreSampledSurfaceCamOurs({}, index, numSampling=None)
            batch0_np['preFullsurfacePCxyzCam'] = tmp['surfacePCxyzCamOurs']
            batch0_np['preFullsurfacePCgradSdfCam'] = tmp['surfacePCnormalCamOurs']
            del tmp

        if 'nearsurface' in datasetConf['samplingMethodList']:
            d = datasetConf['nearsurfaceDeltaRange']
            points = batch0_np['preFullsurfacePCxyzCam']
            normals = batch0_np['preFullsurfacePCgradSdfCam']
            assert datasetConf['numSamplingSufficientNearsurface'] <= points.shape[0]
            if datasetConf['numSamplingSufficientNearsurface'] == points.shape[0]:
                pass
            else:
                points, normals = self.randomlyDrawSample(
                    datasetConf['numSamplingSufficientNearsurface'],
                    points,
                    normals,
                )
            delta = d * (np.random.rand(points.shape[0], ).astype(np.float32) * 2. - 1.)
            points = points + normals * delta[:, None]

            batch0_np['preNearsurfacePCoccfloat'] = (delta > 0).astype(np.float32)
            batch0_np['preNearsurfacePCsdf'] = delta
            batch0_np['preNearsurfacePCxyzCam'] = points
            del points, normals

        return batch0_np

    def _getBasic(self, batch0_np, index):
        batch0_np['winWidth'] = self.winWidth[index]
        batch0_np['winHeight'] = self.winHeight[index]
        batch0_np['focalLengthWidth'] = self.focalLengthWidth[index]
        batch0_np['focalLengthHeight'] = self.focalLengthHeight[index]
        batch0_np['catName'] = self.catNameList[index]
        assert batch0_np['winWidth'] == self.datasetConf['winWidth']
        assert batch0_np['winHeight'] == self.datasetConf['winHeight']
        return batch0_np

    def getRawMesh(self, batch0_np, index):
        catID = self.catIDList[index]
        shaID = self.shaIDList[index]
        camObj2World0 = self.camObj2World[index]
        camWorld2CamTheirs0 = self.camWorld2CamTheirs[index]

        with open(self.projRoot + 'v/misc/shapeNetV2pkl/%s_%s.pkl' % (catID, shaID), 'rb') as f:
            pkl = pickle.load(f)
        vertObj0 = pkl['vertObj0']
        face0 = pkl['face0']
        vertObjHom0 = np.concatenate([vertObj0, np.ones_like(vertObj0[:, :1])], 1)
        vertWorldHom0 = np.matmul(vertObjHom0, camObj2World0.transpose())
        vertCamTheirsHom0 = np.matmul(vertWorldHom0, camWorld2CamTheirs0.transpose())

        batch0_np['rawVertObj'] = vertObj0
        batch0_np['rawFace'] = face0
        batch0_np['rawVertCamTheirs'] = vertCamTheirsHom0[:, :3]
        return batch0_np

    def getRawMeshCamOurs(self, batch0_np, index):
        catID = self.catIDList[index]
        shaID = self.shaIDList[index]
        camR0 = self.camRObj2CamOurs[index]
        camT0 = self.camTObj2CamOurs[index]
        with open(self.projRoot + 'v/R/shapenetv2/%s_%s.pkl' % (catID, shaID), 'rb') as f:
            pkl = pickle.load(f)
        vertObj0 = pkl['vertObj0']
        face0 = pkl['face0']
        vertCamOurs0 = np.matmul(vertObj0, camR0.transpose()) + camT0[None, :]
        batch0_np['vertObj'] = vertObj0
        batch0_np['face'] = face0
        batch0_np['vertCamOurs'] = vertCamOurs0
        return batch0_np

    def getPreSampledSurfaceCamOurs(self, batch0_np, index, numSampling):
        catID = self.catIDList[index]
        shaID = self.shaIDList[index]
        camR0 = self.camRObj2CamOurs[index]
        camT0 = self.camTObj2CamOurs[index]
        with open(self.projRoot + 'v/misc/shapeNetV2surfaceSamplingPkl/%s_%s.pkl' % (catID, shaID), 'rb') as f:
            pkl = pickle.load(f)
        surfacePCxyzObj0 = pkl['surfacePCxyzObj0']
        surfacePCnormalObj0 = pkl['surfacePCnormalObj0']
        if numSampling is not None and numSampling > 0:
            ra = np.random.permutation(surfacePCxyzObj0.shape[0])[:numSampling]
            surfacePCxyzObj0 = surfacePCxyzObj0[ra, :]
            surfacePCnormalObj0 = surfacePCnormalObj0[ra, :]
        surfacePCxyzCamOurs0 = np.matmul(surfacePCxyzObj0, camR0.transpose()) + camT0[None, :]
        surfacePCnormalCamOurs0 = np.matmul(surfacePCnormalObj0, camR0.transpose())
        surfacePCnormalCamOurs0 = np.divide(
            surfacePCnormalCamOurs0,
            np.linalg.norm(surfacePCnormalCamOurs0, ord=2, axis=1)[:, None] + 1.e-6,
        )
        batch0_np['surfacePCxyzCamOurs'] = surfacePCxyzCamOurs0
        batch0_np['surfacePCnormalCamOurs'] = surfacePCnormalCamOurs0
        return batch0_np

    def _getImg(self, batch0_np, index, imageSourceKey):
        sptTag = self.sptTagList[index]
        splitTag = self.splitTagList[index]
        corenetShaH3Tag = self.corenetShaH3TagList[index]
        corenetShaTag = self.corenetShaTagList[index]
        i = dict(np.load(
            self.projRoot + 'remote_fastdata/corenet/data/%s.%s/%s/%s.npz' %
            (sptTag, splitTag, corenetShaH3Tag, corenetShaTag)
        ))[imageSourceKey]
        img0 = np.array(PIL.Image.open(io.BytesIO(i)), dtype=np.float32) / 255.
        batch0_np['img'] = img0.transpose((2, 0, 1))
        assert batch0_np['img'].shape[1] == self.datasetConf['winHeight']
        assert batch0_np['img'].shape[2] == self.datasetConf['winWidth']
        return batch0_np

    def __getitem__(self, indexTrainValTest):
        overfitIndexTvt = self.datasetConf.get('singleSampleOverfittingIndexTvt', 0)

        if self.datasetSplit == 'train':
            index = self.indTrain[indexTrainValTest]
            overfitIndex = self.indTrain[overfitIndexTvt]
        elif self.datasetSplit == 'val':
            index = self.indVal[indexTrainValTest]
            overfitIndex = self.indVal[overfitIndexTvt]
        elif self.datasetSplit == 'test':
            index = self.indTest[indexTrainValTest]
            overfitIndex = self.indTest[overfitIndexTvt]
        elif self.datasetSplit == 'trainval':
            index = self.indTrainval[indexTrainValTest]
            overfitIndex = self.indTrainval[overfitIndexTvt]
        else:
            print('Warning: Since your datasetSplit is %s, you cannot call this function!' %
                  self.datasetSplit)
            raise ValueError

        if self.datasetConf.get('singleSampleOverfittingMode', False):
            index = int(overfitIndex)
        else:
            index = int(index)

        b0np = self.getOneNP(index)
        return b0np
