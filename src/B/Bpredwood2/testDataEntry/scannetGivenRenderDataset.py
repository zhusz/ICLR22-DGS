# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
import numpy as np
import pickle
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from skimage.io import imread
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from skimage.io import imsave
import time
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


class ScannetObjectCache(object):  # OK to be put into each data loader
    def __init__(self, **kwargs):
        self.useful_nyu40IDs = kwargs['useful_nyu40IDs']

        self.scannetObjectCache = {}

    def call_cache_scannet_object(self, **kwargs):
        houseID0 = kwargs['houseID0']
        verbose = kwargs['verbose']
        assert type(houseID0) is int

        if houseID0 not in self.scannetObjectCache.keys():
            if verbose:
                print('    [CacheScannetDataLoader] Loading objectness for scannet - %d' %
                      houseID0)
            mat = udl('mats_R17b_', 'scannet', houseID0)
            newMat = {
                'objNyu40ID': mat['objNyu40ID'].astype(np.int32),
                'objBoundMinWorld': mat['objBoundMinWorld'].astype(np.float32),
                'objBoundMaxWorld': mat['objBoundMaxWorld'].astype(np.float32),
                # 'objCentroidWorld': mat['objCentroidWorld'].astype(np.float32),
            }
            newMat['objCentroidWorld'] = 0.5 * (
                newMat['objBoundMinWorld'] + newMat['objBoundMaxWorld'])
            flagObj = np.isin(newMat['objNyu40ID'], self.useful_nyu40IDs)
            self.scannetObjectCache[houseID0] = {k: newMat[k][flagObj] for k in newMat.keys()}
            # box sys:
            # Note the 3D box chosen in different views would have the z axis alinged with the
            # world sys (ungrav direction). For y axis, it is the depth direction
            # x axis: the remaining one.
        return self.scannetObjectCache[houseID0]


class ScannetMeshCache(object):  # Too large, not able to put into each data loader
    def __init__(self, **kwargs):
        self.scannetHouseVertWorldCache = {}

        semanticPIX3D = udl('_Z_semanticPIX3D')
        self.nyu40ToPix3d10 = semanticPIX3D['nyu40ToPix3d10']
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
                facePix3d10ID0 = self.nyu40ToPix3d10[faceNyu40ID0]
                facePix3d10ID0[faceNyu40ID0 == 0] = -1
                faceNyu40ID0[faceNyu40ID0 == 0] = -1
                self.scannetHouseVertWorldCache[houseID0] = {
                    'vertWorld0': mat['vert0'],
                    'face0': mat['face0'],
                    'faceCentroidWorld0': mat['faceCentroid0'],
                    'faceNormalWorld0': mat['faceNormal0'],
                    'faceNyu40ID0': faceNyu40ID0,
                    'facePix3d10ID0': facePix3d10ID0,
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


class ScannetTsdfvoxCache(object):
    def __init__(self):
        self.scannetTsdfvoxCache = {}
        self.projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
        self.tsdf_vox_folder = self.projRoot + 'remote_fastdata/scannet/scannet_cache/all_tsdf_9/'
        assert os.path.isdir(self.tsdf_vox_folder)

    def call_cache_scannet_tsdfvox_0(self, **kwargs):
        scanID = kwargs['scanID']
        assert type(scanID) is str
        verbose = kwargs['verbose']

        if scanID not in self.scannetTsdfvoxCache.keys():
            if verbose:
                print('    [CacheScannetTsdfvox] Loading tsdf vox for scannet - %s' % scanID)
            with open(self.tsdf_vox_folder + '%s/tsdf_info.pkl' % scanID, 'rb') as f:
                tsdf_info = pickle.load(f)
            vol_origin = tsdf_info['vol_origin']
            assert vol_origin.shape == (3, )
            voxel_size = float(tsdf_info['voxel_size'])
            with np.load(self.tsdf_vox_folder + '%s/full_tsdf_layer0.npz' % scanID) as npz:
                arr_0 = npz['arr_0']  # (XYZ volume)
            arr_zyx_0 = arr_0.transpose((2, 1, 0))  # change into ZYX volume
            self.scannetTsdfvoxCache[scanID] = {
                'vol_origin': vol_origin,
                'vol_end': vol_origin + voxel_size * np.array(arr_0.shape, dtype=np.float32),
                'voxel_size': voxel_size,
                'arr_zyx_0': arr_zyx_0,
            }
        return self.scannetTsdfvoxCache[scanID]


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
        # for k in ['focalLengthWidth', 'focalLengthHeight', 'winWidth', 'winHeight']:
        #     # 'fScaleWidth', 'fScaleHeight']:
        #     # These intrinsics settings should be fixed for the whole training process.
        #     assert k in datasetConf
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
        dataset = datasetConf['dataset']
        houseID = int(self.houseIDList[index])
        viewID = int(self.viewIDList[index])

        b0np = dict(
            index=np.array(index, dtype=np.int32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
            dataset=self.dataset,
        )

        # easy fetch
        b0np['EWorld'] = self.EWorld[index, :]
        b0np['LWorld'] = self.LWorld[index, :]
        b0np['UWorld'] = self.UWorld[index, :]
        b0np['houseID'] = self.houseIDList[index].astype(np.float32)
        b0np['viewID'] = self.viewIDList[index].astype(np.float32)
        b0np['scanID'] = self.scanIDList_house[houseID]

        b0np['winWidth'] = np.array(float(datasetConf['winWidth']), dtype=np.float32)
        b0np['winHeight'] = np.array(float(datasetConf['winHeight']), dtype=np.float32)
        b0np['focalLengthWidth'] = np.array(float(datasetConf['focalLengthWidth']), dtype=np.float32)
        b0np['focalLengthHeight'] = np.array(float(datasetConf['focalLengthHeight']), dtype=np.float32)
        b0np['fScaleWidth'] = np.array(float(datasetConf['fScaleWidth']), dtype=np.float32)
        b0np['fScaleHeight'] = np.array(float(datasetConf['fScaleHeight']), dtype=np.float32)

        # easy processing
        b0np['cam'] = ELU02cam0(np.concatenate([
            b0np['EWorld'], b0np['LWorld'], b0np['UWorld'],
        ], 0))

        ifDummy = datasetConf['ifDummy']

        # img crop
        if not ifDummy:
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
        else:
            b0np['imgForUse'] = np.random.rand(3, datasetConf['winHeight'], datasetConf['winWidth']).astype(np.float32)

        # depth crop
        if not ifDummy:
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
        else:
            b0np['depthForUse'] = np.random.rand(1, datasetConf['winHeight'], datasetConf['winWidth']).astype(np.float32)
            b0np['depthMax'] = np.array(6., dtype=np.float32)

        # face flag
        packedFaceFlag0 = udl(
            'mats_R17fsw%.2ffsh%.2f_packedFaceFlag0' % (b0np['fScaleWidth'], b0np['fScaleHeight']),
            b0np['dataset'], int(b0np['index'])
        )
        faceFlagUntrimmed0  = np.unpackbits(packedFaceFlag0, bitorder='big').astype(np.uint8)
        b0np['faceFlagUntrimmed'] = faceFlagUntrimmed0  # spacial collate_fn

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


def collate_fn_scannetGivenRender(data):
    out = {}
    for k in data[0].keys():
        if k == 'faceFlagUntrimmed':
            out[k] = [torch.from_numpy(x[k]) for x in data]
        elif type(data[0][k]) is str:
            out[k] = [x[k] for x in data]
        elif type(data[0][k]) is np.ndarray or type(data[0][k] is np.int32) or (type(data[0][k] is np.float32)):
            out[k] = torch.from_numpy(np.stack([x[k] for x in data], 0))
        else:
            raise NotImplementedError('Unknown type(k): k: %s, type: %s' % (k, type(data[0][k])))
    return out

