# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
import torch
from skimage.io import imread
import os
import numpy as np
import cv2
import time


'''
class HmMeshCacheFromRaw(object):  # From raw means directly loaded from the raw dataset
    # since loading obj_np directly might be slow, future development reserves the right
    # to extend to loaded from pkls.
    def __init__(self, **kwargs):
        self.hmHouseVertWorldCache = {}

        self.zNear = kwargs['zNear']

        A1_house = udl('pkl_A1_', 'hm')
        self.houseShaList = A1_house['houseShaList']

        projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
        self.hm_raw_root = projRoot + 'remote_fastdata/hm/install/'
        assert os.path.isdir(self.hm_raw_root), self.hm_raw_root

    def call_cache_hm_house_vert_world_0(self, **kwargs):
        houseID0 = kwargs['houseID0']
        verbose = kwargs['verbose']
        assert type(houseID0) is int

        if houseID0 not in self.hmHouseVertWorldCache.keys():
            t = time.time()
            vertWorld0, face0 = load_obj_np(
                self.hm_raw_root + '%05d-%s/%s.obj' %
                (houseID0, self.houseShaList[houseID0], self.houseShaList[houseID0]))
            faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]
            tmp0 = np.cross(
                faceVertWorld0[:, 1, :] - faceVertWorld0[:, 0, :],
                faceVertWorld0[:, 2, :] - faceVertWorld0[:, 0, :],
            )
            faceNormalWorld0 = tmp0 / \
                np.clip(
                    np.linalg.norm(
                        tmp0, axis=1, ord=2), a_min=self.zNear, a_max=np.inf)[:, None]
            # Do not use faceCentroid for hm, as the triangles are pretty large and
            #   their centroids are typically meaningless.
            #   Enable it only if you can write reasoning in here.

            if verbose:
                print('    [CacheHM] Loading house vert world for HM - %d' % houseID0)
                print('    [CacheHM Timing] Elapsed Time is %.3f seconds.' % (time.time() - t))

            self.hmHouseVertWorldCache[houseID0] = {
                'vertWorld0': vertWorld0,
                'face0': face0,
                'faceNormalWorld0': faceNormalWorld0,
            }
        return self.hmHouseVertWorldCache[houseID0]
'''


class HmMeshCache(object):
    def __init__(self, **kwargs):
        self.hmHouseVertWorldCache = {}

    def call_cache_hm_house_vert_world_0(self, **kwargs):
        houseID0 = int(kwargs['houseID0'])
        verbose = kwargs['verbose']
        assert type(houseID0) is int

        if houseID0 not in self.hmHouseVertWorldCache.keys():
            t = time.time()
            try:
                R11 = udl('pkls_R11_', 'hm', houseID0)
            except:
                print(houseID0)
                raise ValueError('houseID0: %d' % houseID0)
            if verbose:
                print('    [CacheHM] Loading house vert world for HM - %d' % houseID0)
                print('    [CacheHM Timing] Elapsed Time is %.3f seconds.' % (time.time() - t))
            self.hmHouseVertWorldCache[houseID0] = {
                'vertWorld0': R11['vertWorld0'],
                'face0': R11['face0'],
                'faceNormalWorld0': R11['faceNormalWorld0'],
                'faceArea0': R11['faceArea0'],
            }
        return self.hmHouseVertWorldCache[houseID0]


class HmRenderOmDataset(object):
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
        assert dataset.startswith('hmRenderOm')
        self.dataset = dataset

        # preset
        self.rendering_rgb_root = self.projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'
        self.rendering_depth_zbuffer_root = \
            self.projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'
        self.hm_mesh_root = self.projRoot + 'remote_fastdata/hm/install/'

        # udl
        A1 = udl('pkl_A1_', dataset)
        self.houseIDList = A1['houseIDList']
        self.pointIDList = A1['pointIDList']
        self.viewIDList = A1['viewIDList']
        self.EWorld = A1['EWorld']
        self.LWorld = A1['LWorld']
        self.UWorld = A1['UWorld']
        self.resolution = A1['resolution']
        self.fScaleList = A1['fScaleList']
        self.flagSplit = udl('mat_A01_flagSplit', dataset)
        # cut off samples whose given fScale is not small enough (the given view too small to crop)
        self.flagSplit[np.isin(
            self.houseIDList,
            np.array([70, 71, 89, 90, 449, 
                475, 476, 477, 478, 479, 480, 484, 486, 487, 489, 490,
                491, 492, 509, 510, 511, 512, 513, 518, 519, 520, 521, 559,
                885, 886, 887, 888, 889, 895, 896, 897, 898],
                dtype=self.houseIDList.dtype))] = 0  # Rendering is wrong
        if datasetConf['ifNeedCroppingAugmentation'] == 0:
            pass
        elif datasetConf['ifNeedCroppingAugmentation'] == 1:
            self.flagSplit[self.fScaleList > datasetConf['fScaleWidthDesiredFixed']] = 0
            self.flagSplit[self.fScaleList > datasetConf['fScaleHeightDesiredFixed']] = 0
        else:
            raise NotImplementedError('Unknown ifNeedCroppingAugmentation: %d' % datasetConf['ifNeedCroppingAugmentation'])
        self.indTrain = np.where(self.flagSplit == 1)[0]
        self.mTrain = int(self.indTrain.shape[0])
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.mVal = int(self.indVal.shape[0])

        self.dataset_house = A1['houseDataset']
        dataset_house = self.dataset_house
        assert dataset_house == 'hm'
        A1_house = udl('pkl_A1_', dataset_house)
        self.houseShaList_house = A1_house['houseShaList']
        self.flagSplit_house = udl('mat_A01_flagSplit', dataset_house)

        # datasetCaches
        self.datasetCaches = {}

        # must be defined
        for k in []:
            assert k in datasetConf, k

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
            raise ValueError('This dataset does not have a test set.')
            return self.mTest
        else:
            raise NotImplementedError('Unknown self.datasetSplit: %s' % self.datasetSplit)

    def getOneNP(self, index):
        # lay out the variables
        datasetConf = self.datasetConf
        dataset = datasetConf['dataset']
        houseID = int(self.houseIDList[index])
        houseSha = self.houseShaList_house[houseID]
        pointID = int(self.pointIDList[index])
        viewID = int(self.viewIDList[index])

        b0np = dict(
            index=np.array(index, dtype=np.int32),
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
            dataset=self.dataset,
        )

        ifDummy = datasetConf['ifDummy']

        # easy fetch
        b0np['EWorld'] = self.EWorld[index, :]
        b0np['LWorld'] = self.LWorld[index, :]
        b0np['UWorld'] = self.UWorld[index, :]
        b0np['houseID'] = self.houseIDList[index].astype(np.int32)
        b0np['pointID'] = self.pointIDList[index].astype(np.int32)
        b0np['viewID'] = self.viewIDList[index].astype(np.int32)
        b0np['houseSha'] = self.houseShaList_house[houseID]
        b0np['ifNeedPixelAugmentation'] = np.array(datasetConf['ifNeedPixelAugmentation'], dtype=np.int32)
        b0np['ifNeedCroppingAugmentation'] = np.array(datasetConf['ifNeedCroppingAugmentation'], dtype=np.int32)
        b0np['ifNeedMirrorAugmentation'] = np.array(datasetConf['ifNeedMirrorAugmentation'], dtype=np.int32)

        # augmentation setting
        # - ifNeedPixelAugmentation -> does not change fxywxy or depth0 or boundMinMaxCamWorld, changes img0
        # - ifNeedCroppingAugmentation -> might change everything (including boundMinMaxCamWorld)
        #       If 2
        #       - fScaleWidthDesiredRange: [small, big]
        #       - fScaleHeightDesiredRange: [small, big]
        #       - (small must be smaller than big)
        #       If 1
        #       - fScaleWidthDesiredFixed
        #       - fScaleHeightDesiredFixed
        #       If 0
        #       (No more config requried). Use self.fScaleList[index]. No Cropping
        # - ifNeedMirrorAugmentation -> does not change fxywxy boundMinMaxCamWorld, but changes img0/depth0

        # easy processing
        b0np['cam'] = ELU02cam0(np.concatenate([
            b0np['EWorld'], b0np['LWorld'], b0np['UWorld'],
        ], 0))

        if datasetConf['ifNeedPixelAugmentation'] or \
                datasetConf['ifNeedCroppingAugmentation'] not in [0, 1] or \
                datasetConf['ifNeedMirrorAugmentation']:
            raise NotImplementedError('Not yet implemented for augmentations.')
            # The things that might be involved include -
            #   fxywxy / fScale
            #   imgForUse
            #   depthForUse

        # fxywxy
        winWidth0 = float(datasetConf['winWidth'])
        winHeight0 = float(datasetConf['winHeight'])
        if datasetConf['ifNeedCroppingAugmentation'] == 0:
            fScaleWidth0 = float(self.fScaleList[index])
            fScaleHeight0 = float(self.fScaleList[index])
        elif datasetConf['ifNeedCroppingAugmentation'] == 1:
            fScaleWidth0 = float(datasetConf['fScaleWidthDesiredFixed'])
            fScaleHeight0 = float(datasetConf['fScaleHeightDesiredFixed'])
        else:
            raise NotImplementedError('Unknown ifNeedCroppingAugmentation: %d' % datasetConf['ifNeedCroppingAugmentation'])
        focalLengthWidth0 = fScaleWidth0 * winWidth0 / 2.
        focalLengthHeight0 = fScaleHeight0 * winHeight0 / 2.
        b0np['winWidth'] = np.array(winWidth0, dtype=np.float32)
        b0np['winHeight'] = np.array(winHeight0, dtype=np.float32)
        b0np['fScaleWidth'] = np.array(fScaleWidth0, dtype=np.float32)
        b0np['fScaleHeight'] = np.array(fScaleHeight0, dtype=np.float32)
        b0np['focalLengthWidth'] = np.array(focalLengthWidth0, dtype=np.float32)
        b0np['focalLengthHeight'] = np.array(focalLengthHeight0, dtype=np.float32)
        b0np['fxywxy'] = np.array([
            focalLengthWidth0, focalLengthHeight0, winWidth0, winHeight0,
        ], dtype=np.float32)

        # img
        if ifDummy:
            img0 = (np.random.rand(512, 512, 3).astype(np.float32) * 255.).astype(np.uint8)
        else:
            img0 = imread(
                self.rendering_rgb_root + '%05d-%s/point_%d_view_%d_domain_rgb.png' %
                (houseID, houseSha, pointID, viewID)
            )
        assert img0.dtype == np.uint8
        assert img0.shape == (self.resolution[index], self.resolution[index], 3)
        assert img0.max() > 1.5  # not [0-1]
        assert img0.max() <= 255.5  # not [0 - 65536]
        img0 = img0.astype(np.float32) / 255.
        if datasetConf['ifNeedCroppingAugmentation'] == 0:
            pass
        elif datasetConf['ifNeedCroppingAugmentation'] in [1, 2]:
            assert self.fScaleList[index] <= b0np['fScaleWidth']
            assert self.fScaleList[index] <= b0np['fScaleHeight']
            cropBoundary = int(float(self.resolution[index]) / 2. * (1. - float(self.fScaleList[index]) / float(b0np['fScaleWidth'])))
            img0 = img0[
                cropBoundary:int(self.resolution[index]) - cropBoundary, 
                cropBoundary:int(self.resolution[index]) - cropBoundary,
                :
            ]
        else:
            raise NotImplementedError('Unknown ifNeedCroppingAugmentation: %d' % datasetConf['ifNeedCroppingAugmentation'])
        img0 = cv2.resize(
            img0, (int(b0np['winWidth']), int(b0np['winHeight'])),
            interpolation=cv2.INTER_CUBIC,
        )
        img0 = img0.transpose((2, 0, 1))
        b0np['imgForUse'] = img0

        # depth
        if ifDummy:
            depth0 = (np.random.rand(512, 512).astype(np.float32) * 65535.).astype(np.uint16)
        else:
            depth0 = imread(
                self.rendering_depth_zbuffer_root + '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' %
                (houseID, houseSha, pointID, viewID)
            )
        assert depth0.dtype == np.uint16
        assert depth0.shape == (self.resolution[index], self.resolution[index])
        assert depth0.max() > 300
        assert depth0.min() >= 0
        depth0 = depth0.astype(np.float32) / 512.  # 512 is the magic number for hmRenderOm
        if datasetConf['ifNeedCroppingAugmentation'] == 0:
            pass
        elif datasetConf['ifNeedCroppingAugmentation'] in [1, 2]:
            assert self.fScaleList[index] <= b0np['fScaleWidth']
            assert self.fScaleList[index] <= b0np['fScaleHeight']
            cropBoundary = int(float(self.resolution[index]) / 2. * (1. - float(self.fScaleList[index]) / float(b0np['fScaleWidth'])))
            depth0 = depth0[
                cropBoundary:int(self.resolution[index]) - cropBoundary, 
                cropBoundary:int(self.resolution[index]) - cropBoundary,
            ]
        else:
            raise NotImplementedError('Unknown ifNeedCroppingAugmentation: %d' % datasetConf['ifNeedCroppingAugmentation'])
        depth0[(depth0 == 0) | (depth0 > 20.) | (np.isfinite(depth0) == 0)] = np.nan
        depth0 = cv2.resize(depth0, (
            int(b0np['winWidth']), int(b0np['winHeight'])), interpolation=cv2.INTER_LINEAR)
        b0np['depthForUse'] = depth0[None, :, :]

        # faceFlag
        with np.load(self.projRoot + 'v/R/%s/R17/%08d.npz' % (dataset, index)) as npz:
            packedFaceFlag0 = npz['packedFaceFlag0']
            b0np['boundMinCam'] = npz['minCam0']
            b0np['boundMaxCam'] = npz['maxCam0']
            b0np['boundMinWorld'] = npz['minWorld0']
            b0np['boundMaxWorld'] = npz['maxWorld0']
            b0np['depthMax'] = npz['depthMax0']
        faceFlagUntrimmed0 = np.unpackbits(packedFaceFlag0, bitorder='big').astype(np.uint8)
        b0np['faceFlagUntrimmed'] = faceFlagUntrimmed0  # special collate_fn

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


def collate_fn_hmRenderOm(data):
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


