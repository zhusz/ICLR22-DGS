# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# obtain faceFlag to determine which triangles are within view frustum
# Note that the triangles in hm are pretty large, so you need to
# test all the three vertices of each triangle, and as long as one of them are within
# the view frustum, the triangle would be counted in.
import os
import sys
import torch
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
import pickle
from scipy.spatial import ConvexHull
from BpredwoodA.testDataEntry.hmRenderOmDataset import HmMeshCacheFromRaw
from BpredwoodA.testDataEntry.testDataEntryPool import getTestDataEntryDict
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert
from collections import OrderedDict
from codes_py.toolbox_show_draw.draw_v1 import to_heatmap


def main():
    hm3d_zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'
    hm3d_rgb_raw_root = projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'
    assert os.path.isdir(hm3d_zbuffer_raw_root), hm3d_zbuffer_raw_root
    assert os.path.isdir(hm3d_rgb_raw_root), hm3d_rgb_raw_root
    assert os.path.isdir(projRoot + 'cache/')
    dataset = os.path.realpath(__file__).split('/')[-2]
    visualDir = projRoot + 'cache/dataset/%s/R17/' % dataset
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 100, 'R17')

    # indChosen_val = getTestDataEntryDict(wishedTestDataNickName=['hmOfficialValSplit10'])['hmOfficialValSplit10']['indChosen']

    A1 = udl('pkl_A1_', dataset)
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    EWorld = A1['EWorld']
    LWorld = A1['LWorld']
    UWorld = A1['UWorld']
    resolution = A1['resolution']
    fScaleList = A1['fScaleList']
    m = A1['m']
    dataset_house = A1['houseDataset']

    flagSplit = udl('_A01_flagSplit', dataset)

    A1_house = udl('pkl_A1_', dataset_house)
    houseShaList = A1_house['houseShaList']

    cudaDevice = 'cuda:0'
    zNear = 1.e-6
    slacked = 1.1
    hmMeshCache = HmMeshCacheFromRaw(zNear=zNear)

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2
    indChosen = list(range(j1, j2))
    indVisChosen = [0, 1, 2, 3, 4, 5, 6, 1000, 10000, 100000, 1000000, 5000000, 8470854]

    for j in indChosen:

        if flagSplit[j] <= 0:
            continue

        # if (flagSplit[j] > 1) and (j not in indChosen_val):
        if (flagSplit[j] != 2):
            continue

        if j < 10 or j % 1 == 0:
            print('Processing R17_faceFlag for %s: %d (J1 = %d, J2 = %d)' %
                  (dataset, j, j1, j2))

        # the first val case: 8470854

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
        tmp0 = hmMeshCache.call_cache_hm_house_vert_world_0(houseID0=houseID0, verbose=1)
        vertWorld0 = torch.from_numpy(tmp0['vertWorld0']).float().to(cudaDevice)
        face0 = torch.from_numpy(tmp0['face0']).int().to(cudaDevice)
        nFace = int(face0.shape[0])
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

        depthMax0 = np.quantile(depth0[np.isfinite(depth0)], 0.98)
        depthMax0 = depthMax0 * slacked
        depth0[depth0 > depthMax0] = np.nan
        tmp0 = depthMap2mesh(depth0, fScale0, fScale0, cutLinkThre=None)
        depthPCxyzCam0 = torch.from_numpy(tmp0['vertCam0']).float().to(cudaDevice)
        depthPCxyzWorld0 = torch.matmul(
            depthPCxyzCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse

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
        faceFlag0 = torch.any(vertFlag0[face0.long()], dim=1).detach().cpu().numpy().astype(bool)

        packedFaceFlag0 = np.packbits(faceFlag0.astype(bool), bitorder='big')
        np.savez_compressed(
            projRoot + 'v/R/%s/R17/%08d.npz' % (dataset, j),
            packedFaceFlag0=packedFaceFlag0,
            minCam0=minCam0.detach().cpu().numpy(), maxCam0=maxCam0.detach().cpu().numpy(),
            minWorld0=minWorld0.detach().cpu().numpy(), maxWorld0=maxWorld0.detach().cpu().numpy(),
            depthMax0=float(depthMax0),
        )
        # with open(projRoot + 'v/R/%s/R17/%08d.pkl' % (dataset, j), 'wb') as f:
        #     pickle.dump({
        #         'packedFaceFlag0': packedFaceFlag0,
        #     }, f)

        if j in indVisChosen:
            vertFlag0 = vertFlag0.detach().cpu().numpy()
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
                              '%05d-%s)' % (dataset, j, dataset_house, houseID0, houseSha),
                subMessage='',
            )

            vertRgb0 = 0.2 * np.ones((vertCam0.shape[0], 3), dtype=np.float32)
            vertRgb0[vertFlag0 > 0, 0] = 0.6
            vertRgb0[vertFlag0 == 0, 2] = 0.6
            dumpPly(
                visualDir + '%s_%08d_house_%05d_vertFlag_cam.ply' % (dataset, j, houseID0),
                vertCam0, face0, vertRgb0,
            )
            faceRgb0 = 0.2 * np.ones((face0.shape[0], 3), dtype=np.float32)
            faceRgb0[faceFlag0 > 0, 0] = 0.6
            faceRgb0[faceFlag0 == 0, 2] = 0.6
            dumpPly2(
                visualDir + '%s_%08d_house_%05d_faceFlag_cam.ply' % (dataset, j, houseID0),
                vertCam0, face0, faceRgb0,
            )
            dumpPlyPointCloud(
                visualDir + '%s_%08d_house_%05d_depthPointCloud_cam.ply' %
                (dataset, j, houseID0),
                depthPCxyzCam0,
            )
            tmp0, tmp1 = trimVert(vertCam0, face0[faceFlag0 > 0])
            dumpPly(
                visualDir + '%s_%08d_house_%05d_remaining_cam.ply' % (dataset, j, houseID0),
                tmp0, tmp1,
            )


if __name__ == '__main__':
    main()

