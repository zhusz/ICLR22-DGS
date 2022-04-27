# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

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
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow, to_heatmap
from skimage.io import imsave
import cv2
import os


bt = lambda s: s[0].upper() + s[1:]


def rollup3DBoxCorners(minBound, maxBound):
    assert len(minBound.shape) == 2 and minBound.shape[1] == 3
    assert minBound.shape == maxBound.shape
    nBox = minBound.shape[0]

    cornerBound = np.zeros((nBox, 8, 3), dtype=np.float32)
    cornerBound[:, 0, :] = np.stack([minBound[:, 0], minBound[:, 1], minBound[:, 2]], 1)
    cornerBound[:, 1, :] = np.stack([maxBound[:, 0], minBound[:, 1], minBound[:, 2]], 1)
    cornerBound[:, 2, :] = np.stack([minBound[:, 0], maxBound[:, 1], minBound[:, 2]], 1)
    cornerBound[:, 3, :] = np.stack([maxBound[:, 0], maxBound[:, 1], minBound[:, 2]], 1)
    cornerBound[:, 4, :] = np.stack([minBound[:, 0], minBound[:, 1], maxBound[:, 2]], 1)
    cornerBound[:, 5, :] = np.stack([maxBound[:, 0], minBound[:, 1], maxBound[:, 2]], 1)
    cornerBound[:, 6, :] = np.stack([minBound[:, 0], maxBound[:, 1], maxBound[:, 2]], 1)
    cornerBound[:, 7, :] = np.stack([maxBound[:, 0], maxBound[:, 1], maxBound[:, 2]], 1)
    return cornerBound


class ScannetMeshCache(object):
    def __init__(self):
        self.scannetHouseVertWorldCache = {}

        self.flagSplit = udl('_A01_flagSplit', 'scannet')

    def call_cache_scannet_house_vert_world_0(self, **kwargs):
        houseID0 = kwargs['houseID0']
        verbose = kwargs['verbose']
        assert type(houseID0) is int

        if houseID0 not in self.scannetHouseVertWorldCache.keys():
            if verbose:
                print('    [CacheScannet] Loading house vert world for scannet - %d' % houseID0)
            mat = udl('mats_R17_', 'scannet', houseID0)
            if self.flagSplit[houseID0] in [1, 2]:
                faceNyu40ID0 = mat['faceNyu40ID0']
                faceNyu40ID0[faceNyu40ID0 > 40] = 0
                faceNyu40ID0[faceNyu40ID0 == 0] = -1
                self.scannetHouseVertWorldCache[houseID0] = {
                    'vertWorld0': mat['vert0'],
                    'face0': mat['face0'],
                    'faceCentroidWorld0': mat['faceCentroid0'],
                    'faceNormalWorld0': mat['faceNormal0'],
                    'faceNyu40ID0': faceNyu40ID0,
                }
            elif self.flagSplit[houseID0] in [3]:
                self.scannetHouseVertWorldCache[houseID0] = {
                    'vertWorld0': mat['vert0'],
                    'face0': mat['face0'],
                    'faceCentroidWorld0': mat['faceCentroid0'],
                    'faceNormalWorld0': mat['faceNormal0']
                }
            else:
                raise NotImplementedError('Unknown flagSplit %d' % self.flagSplit[houseID0])
        return self.scannetHouseVertWorldCache[houseID0]


class ScannetGivenRenderDataset(object):
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

        # datasetCaches
        self.datasetCaches = {}

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
                      'color/%d.jpg' % viewID, as_gray=False)
        assert len(img0.shape) == 3 and img0.shape[2] == 3
        assert 'uint8' in str(img0.dtype)
        assert img0.max() > 50 and img0.min() >= 0 and img0.max() <= 255
        img0 = img0.astype(np.float32) / 255.
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
                                'depth/%d.png' % viewID)  # , required_dtype='uint16', required_nc=1)
        assert len(depthOriginal0.shape) == 2
        assert 'uint16' in str(depthOriginal0.dtype)
        assert depthOriginal0.max() == 0 or depthOriginal0.max() > 300  # uint16
        assert depthOriginal0.min() >= 0
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

