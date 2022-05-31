# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a session written in hm, but mainly serving for hmRenderOm

# Two engineering modification required:
# 1. convex hull is too slow and cannot finish within reasonable time, so change back to
#      slacked view frustum + world sys min/max bound. See R17 scannetGivenRender
# 2. Unoptimized storage (np.compress_npz on packedbit requires 50G with 1000W files)
#      Change to per_house storing (so with much less files).
#      Move this file to hm...

# obtain faceFlag to determine which triangles are within view frustum
# Note that the triangles in hm are pretty large, so you need to
# test all the three vertices of each triangle, and as long as one of them are within
# the view frustum, the triangle would be counted in.

# Also dump flag1234 for determining whether a rendering is valid or not.
# # 1. mean depth >= 1.0 (rule out E problem looking at the wall)
# meanDepth = np.mean(depthConventional0[np.isfinite(depthConventional0)])
# flag1 = meanDepth >= 1.0
#
# # 2. min depth >= 0.2 (rule out E problem looking at the wall)
# minDepth = np.min(depthConventional0[np.isfinite(depthConventional0)])
# flag2 = minDepth >= 0.2
#
# # 3. percentageFinite >= 0.70 (rule out L problem looking at outside of the room)
# flagFinite0 = np.isfinite(depthConventional0)
# percentageFinite = float(flagFinite0.sum()) / np.prod(flagFinite0.shape)
# flag3 = percentageFinite >= 0.70
#
# # 4. depthConventional0 < depthClockwise0 percentage > 0.8
# t = (depthConventional0 < depthClockwise0).sum()
# f = (depthConventional0 > depthClockwise0).sum()
# flag4 = (float(t) / float(t + f) > 0.8)
#
# Now flag4 is always true.
# flag1234 is all zero for the case where houseID0 is 887. (inherented from hmRenderOm-A1)

# As long as there is one zero among flag1234, the faceFlag0 would be all-zero.

import torch
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np, dumpPly, dumpPly2, dumpPlyPointCloud
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from skimage.io import imread
from UDLv3 import udl
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
import numpy as np
from scipy.spatial import ConvexHull
from BpredwoodA.testDataEntry.hmRenderOmDataset import HmMeshCacheFromRaw
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert
from collections import OrderedDict
from codes_py.toolbox_show_draw.draw_v1 import to_heatmap
import pickle


def mainGPU():
    hm3d_zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'
    hm3d_rgb_raw_root = projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'
    assert os.path.isdir(hm3d_zbuffer_raw_root), hm3d_zbuffer_raw_root
    assert os.path.isdir(hm3d_rgb_raw_root), hm3d_rgb_raw_root
    assert os.path.isdir(projRoot + 'cache/')

    # dataset_self and dataset_rendering
    dataset_self = 'hm'
    assert dataset_self == os.path.realpath(__file__).split('/')[-2]
    dataset_rendering = 'hmRenderOm'
    visualDir = projRoot + 'cache/dataset/%s/R7/' % dataset_self
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 100, 'R7')

    A1 = udl('pkl_A1_', dataset_rendering)
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    EWorld = A1['EWorld']
    LWorld = A1['LWorld']
    UWorld = A1['UWorld']
    resolution = A1['resolution']
    fScaleList = A1['fScaleList']
    m = A1['m']
    assert dataset_self == A1['houseDataset']

    A1_house = udl('pkl_A1_', dataset_self)
    houseShaList = A1_house['houseShaList']

    j1_self = int(os.environ['J1'])
    j2_self = int(os.environ['J2'])
    assert 0 <= j1_self < j2_self
    indChosen = list(range(j1_self, j2_self))
    indVisChosen = [0, 1, 850]

    zNear = 1.e-6
    slacked = 1.05
    hmMeshCache = HmMeshCacheFromRaw(zNear=zNear)
    cudaDevice = 'cuda:0'

    for j_self in indChosen:
        if not os.path.isfile(projRoot + 'v/R/%s/R7/%08d.npz' % (dataset_self, j_self)):
            print('Processing R7_hmRenderingOmFaceFlag for %s: %d (J1 = %d, j2_self = %d)' %
                  (dataset_self, j_self, j1_self, j2_self))
        else:
            print('Skipping R7_hmRenderingOmFaceFlag for %s: %d (J1 = %d, j2_self = %d)' %
                  (dataset_self, j_self, j1_self, j2_self))
            continue

        '''
        if (60 <= j_self < 70) or (120 <= j_self < 130) or (190 <= j_self < 200) \
                or (290 <= j_self < 300):
            continue
        '''

        # input and output
        tmp0 = hmMeshCache.call_cache_hm_house_vert_world_0(houseID0=j_self, verbose=1)
        vertWorld0 = torch.from_numpy(tmp0['vertWorld0']).float().to(cudaDevice)
        face0 = torch.from_numpy(tmp0['face0']).int().to(cudaDevice)
        nFace = int(face0.shape[0])
        indChosen_rendering = np.where(houseIDList == j_self)[0]
        assert np.all(indChosen_rendering[1:] == indChosen_rendering[:-1] + 1)
        startingInd = int(indChosen_rendering[0])
        nRendering = int(indChosen_rendering.shape[0])
        faceFlag = np.zeros((nRendering, nFace), dtype=bool)
        flag123 = np.zeros((nRendering, 3), dtype=bool)
        # store:
        # packedFaceFlag (reshaped from (nRendering, nFace)), startingInd, nRendering, nFace)

        for i, j in enumerate(indChosen_rendering):
            if i < 10 or i % 1 == 0:
                print('    j_self = %d, i = %d, j = %d, j_self = %d, nRendering = %d' %
                      (j_self, i, j, j_self, nRendering))

            houseID0 = int(houseIDList[j])
            houseSha = houseShaList[houseID0]
            pointID0 = int(pointIDList[j])
            viewID0 = int(viewIDList[j])
            fScale0 = float(fScaleList[j])
            cam0 = ELU02cam0(np.concatenate([
                EWorld[j, :], LWorld[j, :], UWorld[j, :]
            ], 0))
            cam0 = torch.from_numpy(cam0).float().to(cudaDevice)
            camR0 = cam0[:3, :3]
            camT0 = cam0[:3, 3]
            vertCam0 = torch.matmul(vertWorld0, camR0.T) + camT0[None, :]
            vertCamPersp0 = torch.stack([
                fScale0 * vertCam0[:, 0] /
                    torch.clamp(vertCam0[:, 2], min=zNear),
                fScale0 * vertCam0[:, 1] /
                    torch.clamp(vertCam0[:, 2], min=zNear),
                vertCam0[:, 2],
            ], 1)

            # depthMap for bound
            depth0 = imread(
                hm3d_zbuffer_raw_root + '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' %
                (houseID0, houseSha, pointID0, viewID0)).astype(np.float32) / 512.
            depth0[np.isfinite(depth0) == 0] = np.nan
            depth0[depth0 < 0.05] = np.nan
            depth0[depth0 > 10] = np.nan

            # flag1234
            finite_depth = depth0[np.isfinite(depth0)]
            if j_self != 887 and finite_depth.shape[0] > 0:
                flag1 = np.mean(finite_depth) >= 1.0
                flag2 = np.min(finite_depth) >= 0.2
                flag3 = (float(finite_depth.shape[0]) / float(np.prod(depth0.shape))) >= 0.7
                flag4 = True
            else:
                flag1, flag2, flag3, flag4 = False, False, False, False
            flag123[i, 0], flag123[i, 1], flag123[i, 2] = flag1, flag2, flag3
            if not (flag1 and flag2 and flag3 and flag4):
                continue  # leave faceFlag0 to be all zeros
            # Note you still need to judge this flag1234 separately in hmRenderOm-A01.

            depthMax0 = np.quantile(depth0[np.isfinite(depth0)], 0.98)
            depth0[depth0 > depthMax0] = np.nan
            tmp0 = depthMap2mesh(depth0, fScale0, fScale0, cutLinkThre=None)
            depthPCxyzCam0 = torch.from_numpy(tmp0['vertCam0']).float().to(cudaDevice)
            depthPCxyzWorld0 = torch.matmul(
                depthPCxyzCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse

            '''
            minCam0, maxCam0 = depthPCxyzCam0.min(0).values, depthPCxyzCam0.max(0).values
            minWorld0, maxWorld0 = depthPCxyzWorld0.min(0).values, depthPCxyzWorld0.max(0).values
            '''
            minCam0 = torch.quantile(depthPCxyzCam0, 0.02, dim=0)
            maxCam0 = torch.quantile(depthPCxyzCam0, 0.98, dim=0)
            minWorld0 = torch.quantile(depthPCxyzWorld0, 0.02, dim=0)
            maxWorld0 = torch.quantile(depthPCxyzWorld0, 0.98, dim=0)

            minCam0 = slacked * minCam0 + (1 - slacked) * maxCam0
            maxCam0 = slacked * maxCam0 + (1 - slacked) * minCam0
            minWorld0 = slacked * minWorld0 + (1 - slacked) * maxWorld0
            maxWorld0 = slacked * maxWorld0 + (1 - slacked) * minWorld0

            # from vertFlag to faceFlag
            vertFlag0 = (
                (torch.abs(vertCamPersp0[:, 0]) < slacked) &
                (torch.abs(vertCamPersp0[:, 1]) < slacked) &
                (vertCamPersp0[:, 2] > zNear) &
                (torch.all(vertCam0 > minCam0[None, :], dim=1)) &
                (torch.all(vertCam0 < maxCam0[None, :], dim=1)) &
                (torch.all(vertWorld0 > minWorld0[None, :], dim=1)) &
                (torch.all(vertWorld0 < maxWorld0[None, :], dim=1))
            )
            faceFlag0 = torch.any(vertFlag0[face0.long()], dim=1)
            faceFlag[i, :] = faceFlag0.detach().cpu().numpy()

            if (j_self in indVisChosen) and (i in [100, 200, 300, 400]):
                vertFlag0 = vertFlag0.detach().cpu().numpy()
                faceFlag0 = faceFlag0.detach().cpu().numpy()
                vertCam0 = vertCam0.detach().cpu().numpy()
                face0 = face0.detach().cpu().numpy()

                summary0 = OrderedDict([])
                txt0 = []
                brInds = [0, ]
                omnidata_img = imread(
                    hm3d_rgb_raw_root + '%05d-%s/point_%d_view_%d_domain_rgb.png' %
                    (houseID0, houseSha, pointID0, viewID0)
                )
                summary0['omnidata_img'] = omnidata_img
                txt0.append('')
                omnidata_depth = depth0
                summary0['omnidata_depth_zbuffer'] = to_heatmap(omnidata_depth, vmin=0.5, vmax=6, cmap='inferno')
                txt0.append('')
                brInds.append(len(summary0))
                htmlStepper.step2(
                    summary0, txt0, brInds,
                    headerMessage='Dataset: %s, Index = %d (houseDataset: %s, '
                                  '%05d-%s)' % (dataset_rendering, j, dataset_self, houseID0, houseSha),
                    subMessage='',
                )

                vertRgb0 = 0.2 * np.ones((vertCam0.shape[0], 3), dtype=np.float32)
                vertRgb0[vertFlag0 > 0, 0] = 0.6
                vertRgb0[vertFlag0 == 0, 2] = 0.6
                dumpPly(
                    visualDir + '%s_%08d_house_%05d_vertFlag_cam.ply' % (dataset_rendering, j, houseID0),
                    vertCam0, face0, vertRgb0,
                )
                faceRgb0 = 0.2 * np.ones((face0.shape[0], 3), dtype=np.float32)
                faceRgb0[faceFlag0 > 0, 0] = 0.6
                faceRgb0[faceFlag0 == 0, 2] = 0.6
                dumpPly2(
                    visualDir + '%s_%08d_house_%05d_faceFlag_cam.ply' % (dataset_rendering, j, houseID0),
                    vertCam0, face0, faceRgb0,
                )
                dumpPlyPointCloud(
                    visualDir + '%s_%08d_house_%05d_depthPointCloud_cam.ply' %
                    (dataset_rendering, j, houseID0),
                    depthPCxyzCam0,
                )
                tmp0, tmp1 = trimVert(vertCam0, face0[faceFlag0 > 0])
                dumpPly(
                    visualDir + '%s_%08d_house_%05d_remaining_cam.ply' % (dataset_rendering, j, houseID0),
                    tmp0, tmp1,
                )

                vertCam0 = torch.from_numpy(vertCam0).float().to(cudaDevice)
                face0 = torch.from_numpy(face0).int().to(cudaDevice)

        packedFaceFlag = np.packbits(faceFlag.reshape((nRendering * nFace)), bitorder='big')
        packedFlag123 = np.packbits(flag123.reshape((nRendering * 3)), bitorder='big')
        # Only for reference purpose. You still need to judge it aga

        np.savez_compressed(
            projRoot + 'v/R/%s/R7/%08d.npz' % (dataset_self, j_self),
            packedFaceFlag=packedFaceFlag,
            packedFlag123=packedFlag123,
            startingInd=startingInd,
            nRendering=nRendering,
            nFace=nFace,
        )


if __name__ == '__main__':
    mainGPU()
