# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
from configs_registration import getConfigGlobal
from datasets_registration import datasetDict
import time
import os
import sys
import numpy as np
import trimesh

from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from codes_py.np_ext.np_image_io_v1 import imread
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from codes_py.template_visualStatic.visualStatic_v1 import TemplateVisualStatic
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert, combineMultiShapes_withVertRgb, \
    addPointCloudToMesh
from codes_py.toolbox_show_draw.draw_v1 import drawBoxXDXD
from codes_py.toolbox_3D.view_query_generation_v1 import gen_viewport_query_given_bound
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, camSys2CamPerspSys0, camPerspSys2CamSys0
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow
from codes_py.toolbox_show_draw.show3D_v1 import showPoint3D4
from codes_py.toolbox_3D.rotations_v1 import getRotationMatrixBatchNP
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP, vertInfo2faceVertInfoTHGPU
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2, dumpPlyPointCloud
from codes_py.np_ext.np_image_io_v1 import imread
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from codes_py.toolbox_3D.draw_mesh_v1 import pickInitialObservationPoints, getUPick0List
from codes_py.toolbox_3D.self_sampling_v1 import mesh_weighted_sampling_given_normal
from codes_py.toolbox_3D.benchmarking_v1 import packageCDF1, packageDepth, affinePolyfitWithNaN
from pytorch3d.ops.knn import knn_points
import cv2
import math


class SubmissionTimeScannetMeshCache(object):
    def __init__(self):
        self.scannetHouseVertWorldCache = {}

    def call_cache_scannet_house_vert_world_0(self, **kwargs):
        projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
        houseID0 = kwargs['houseID0']
        scannetFile = kwargs['scannetFile']
        scannetScanID = kwargs['scannetScanID']
        original_dataset_root = kwargs.get(
            'original_dataset_root', projRoot + 'remote_fastdata/scannet/')
        verbose = kwargs['verbose']
        assert type(houseID0) is int

        if houseID0 not in self.scannetHouseVertWorldCache.keys():
            if verbose:
                print('    [CacheScannet] Loading house vert world for scannet - %d' % houseID0)
            ply = trimesh.load(original_dataset_root + scannetFile + scannetScanID + '_vh_clean_2.ply')
            vertWorld0 = np.array(ply.vertices, dtype=np.float32)
            face0 = np.array(ply.faces, dtype=np.int32)
            faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]
            faceCentroidWorld0 = faceVertWorld0.mean(1)
            tmp0 = np.cross(faceVertWorld0[:, 1, :] - faceVertWorld0[:, 0, :],
                            faceVertWorld0[:, 2, :] - faceVertWorld0[:, 0, :])
            faceNormalWorld0 = np.divide(tmp0, np.linalg.norm(tmp0, axis=1, ord=2)[:, None])
            self.scannetHouseVertWorldCache[houseID0] = {
                'vertWorld0': vertWorld0,
                'face0': face0,
                'faceCentroidWorld0': faceCentroidWorld0,
                'faceNormalWorld0': faceNormalWorld0,
            }
        return self.scannetHouseVertWorldCache[houseID0]


class SubmissionTimeScannetGivenRenderDataset(object):
    # potentially being put into data loader with multiple replication in the CPU memory
    # So make sure no caching things in here
    def __init__(self, datasetConf, **kwargs):
        # basics
        self.meta = {}

        # kwargs
        self.projRoot = kwargs['projRoot']
        self.datasetSplit = kwargs['datasetSplit']
        R = kwargs['R']

        # datasetConf
        self.datasetConf = datasetConf

        # layout variables
        dataset = datasetConf['dataset']
        assert dataset.startswith('scannetGivenRender')
        self.dataset = dataset

        # preset
        self.cache_dataset_root = self.projRoot + 'remote_fastdata/scannet/scannet_cache/'
        self.original_dataset_root = self.projRoot + 'remote_fastdata/scannet/'

        # udl
        A1 = udl('pkl_A1_', dataset)
        self.dataset_house = A1['houseDataset']
        dataset_house = self.dataset_house
        self.fxywxyColor = A1['fxywxyColor']
        self.fxywxyDepth = A1['fxywxyDepth']
        self.ppxyColor = A1['ppxyColor']
        self.ppxyDepth = A1['ppxyDepth']
        self.houseIDList = A1['houseIDList']
        self.viewIDList = A1['viewIDList']
        A1b = udl('_A1b_', dataset)
        self.EWorld = A1b['EWorld']
        self.LWorld = A1b['LWorld']
        self.UWorld = A1b['UWorld']
        A1_house = udl('_A1_', dataset_house)
        self.scanIDList_house = A1_house['scanIDList']
        self.fileList_house = A1_house['fileList']
        A01 = udl('_A01_', dataset)
        self.flagSplit = A01['flagSplit']
        self.indTrain = np.where(self.flagSplit == 1)[0]
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.mTrain = int(len(self.indTrain))
        self.mVal = int(len(self.indVal))
        self.mTest = int(len(self.indTest))
        A01_house = udl('_A01_', dataset_house)
        self.flagSplit_house = A01_house['flagSplit']

        # must be defined
        for k in ['focalLengthWidth', 'focalLengthHeight', 'winWidth', 'winHeight']:
            # 'fScaleWidth', 'fScaleHeight']:
            # These intrinsics settings should be fixed for the whole training process.
            assert k in datasetConf
        # assert datasetConf['fScaleWidth'] == \
        #     2 * datasetConf['focalLengthWidth'] / datasetConf['winWidth']
        # assert datasetConf['fScaleHeight'] == \
        #     2 * datasetConf['focalLengthHeight'] / datasetConf['winHeight']
        # fScale = 2 * focalLengthWidth / winWidth == 2 * focalLengthHeight / winHeight
        # fScale changes, fov changes.
        # If you crop (no resize), focalLength is unchanged, but winSize / fScale / fov are changed.
        # If you resize (no crop), fScale / fov are unchanged, but focalLength / winSize are changed.
        # If you resize and crop to the original size, winSize is unchanged,
        #   but focalLength / fScale / fov are changed.

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
        houseID = int(self.houseIDList[index])
        viewID = int(self.viewIDList[index])

        b0np = dict(
            index=np.array(index, dtype=np.float32),
            did=np.array(did, dtype=np.float32),
            datasetID=np.array(datasetID, dtype=np.float32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
        )

        # easy fetch
        b0np['EWorld'] = self.EWorld[index, :]
        b0np['LWorld'] = self.LWorld[index, :]
        b0np['UWorld'] = self.UWorld[index, :]
        b0np['houseID'] = self.houseIDList[index].astype(np.float32)
        b0np['viewID'] = self.viewIDList[index].astype(np.float32)

        # easy processing
        b0np['cam'] = ELU02cam0(np.concatenate([
            b0np['EWorld'], b0np['LWorld'], b0np['UWorld'],
        ], 0))

        # img crop
        img0 = imread(self.cache_dataset_root + self.fileList_house[houseID] +
                      'color/%d.jpg' % viewID)
        # your resizing scaling ratio is pre-determined here.
        predeterminedResizingScaleOnWidth = datasetConf['focalLengthWidth'] / \
            self.fxywxyColor[index, 0]
        predeterminedResizingScaleOnHeight = datasetConf['focalLengthHeight'] / \
            self.fxywxyColor[index, 1]
        cropImg0 = croppingCxcyrxry0(
            img0, [
                self.ppxyColor[index, 0],
                self.ppxyColor[index, 1],
                datasetConf['winWidth'] / predeterminedResizingScaleOnWidth / 2.,
                datasetConf['winHeight'] / predeterminedResizingScaleOnHeight / 2.,
            ],
            padConst=0,
        )
        b0np['imgForUse'] = cv2.resize(
            cropImg0, (datasetConf['winWidth'], datasetConf['winHeight']),
            interpolation=cv2.INTER_CUBIC,
        ).transpose((2, 0, 1))
        del predeterminedResizingScaleOnWidth, predeterminedResizingScaleOnHeight

        # depth crop
        depthOriginal0 = imread(self.cache_dataset_root + self.fileList_house[houseID] +
                                'depth/%d.png' % viewID, required_dtype='uint16', required_nc=1)
        depth0 = depthOriginal0.copy()
        assert np.nanmin(depth0) >= 0
        maskUnknown0 = (depth0 == 0)
        depth0 = depth0.astype(np.float32) / 1000.
        depth0[maskUnknown0] = np.nan
        assert np.nanmin(depth0) >= 0
        predeterminedResizingScaleOnWidth = datasetConf['focalLengthWidth'] / \
            self.fxywxyDepth[index, 0]
        predeterminedResizingScaleOnHeight = datasetConf['focalLengthHeight'] / \
            self.fxywxyDepth[index, 1]
        cropDepth0 = croppingCxcyrxry0(
            depth0, [
                self.ppxyDepth[index, 0],
                self.ppxyDepth[index, 1],
                datasetConf['winWidth'] / predeterminedResizingScaleOnWidth / 2.,
                datasetConf['winHeight'] / predeterminedResizingScaleOnHeight / 2.,
            ],
            padConst=np.nan,
        )
        assert np.nanmin(cropDepth0) >= 0
        depthForUse0 = cv2.resize(
            cropDepth0, (datasetConf['winWidth'], datasetConf['winHeight']),
            interpolation=cv2.INTER_LINEAR,
        )[None, :, :]
        assert np.nanmin(depthForUse0) >= 0
        b0np['depthForUse'] = depthForUse0
        b0np['depthMax'] = depthForUse0[np.isfinite(depthForUse0)].max()
        del predeterminedResizingScaleOnWidth, predeterminedResizingScaleOnHeight

        return b0np

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
