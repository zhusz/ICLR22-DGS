# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# bash for parallel: for g in {0..53}; do J1=$((g*50000)) J2=$(((g+1)*50000)) op R17.py & done
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from collections import OrderedDict
from codes_py.np_ext.mat_io_v1 import pLoadMat
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from skimage.io import imread
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from codes_py.np_ext.mat_io_v1 import pSaveMat
from codes_py.toolbox_3D.mesh_surgery_v1 import create_cuboid_mesh, \
    combineMultiShapes_withVertRgb
from matplotlib.cm import get_cmap
import pyrender
import cv2
import numpy as np
import time
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly2
np.set_printoptions(suppress=True)


def _call_cache_scannet_house_vert_world_0(**kwargs):
    houseID0 = kwargs['houseID0']
    meta = kwargs['meta']
    projRoot = kwargs['projRoot']
    scannetScanID = kwargs['scannetScanID']
    assert type(houseID0) is int
    if houseID0 not in meta['houseVertWorldCache']['scannet'].keys():
        print('    [CacheScannet] Loading house vert world for scannet - %d' % houseID0)
        t = time.time()
        # mat = pLoadMat(projRoot + 'remote_fastdata/cache/scannet/detailedMesh/%s.mat' % scannetScanID)
        mat = udl('mats_R17_', 'scannet', houseID0)
        print('Elapsed Time is: %.5f seconds.' % (time.time() - t))
        meta['houseVertWorldCache']['scannet'][houseID0] = {
            'vertWorld0': mat['vert0'],
            'face0': mat['face0'],
            'faceCentroidWorld0': mat['faceCentroid0'],
            'faceNormalWorld0': mat['faceNormal0'],
            'faceNyu40ID0': mat.get('faceNyu40ID0', None),
            'faceObjID0': mat.get('faceObjID0', None),
        }
    return meta['houseVertWorldCache']['scannet'][houseID0]


def fix_pose(pose):
    # 3D Rotation about the x-axis.
    t = np.pi
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R
    return pose @ axis_transform


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    cache_dataset_root = projRoot + 'remote_fastdata/scannet/scannet_cache/'
    visualDir = projRoot + 'cache/dataset/scannetGivenRender/R17/'

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    indChosen = list(range(j1, j2))

    # set your set
    focalLengthWidth = 230.4
    focalLengthHeight = 230.4
    winWidth = 256
    winHeight = 192

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    dataset_house = A1['houseDataset']
    houseIDList = A1['houseIDList']
    viewIDList = A1['viewIDList']
    fxywxyColor = A1['fxywxyColor']
    fxywxyDepth = A1['fxywxyDepth']
    ppxyColor = A1['ppxyColor']
    ppxyDepth = A1['ppxyDepth']
    A1b = udl('_A1b_', dataset)
    E = A1b['EWorld']
    L = A1b['LWorld']
    U = A1b['UWorld']
    flagSplit = udl('_A01_flagSplit', dataset)
    A1_house = udl('pkl_A1_', dataset_house)
    scanIDList_house = A1_house['scanIDList']
    fileList_house = A1_house['fileList']

    cam = ELU2cam(np.concatenate([E, L, U], 1))
    fScaleWidth = 2. * focalLengthWidth / winWidth
    fScaleHeight = 2. * focalLengthHeight / winHeight
    indVisChosen = list(range(5))
    meta = {'houseVertWorldCache': {'scannet': {}}}
    zNear = 1.e-6
    slacked = 1.05
    print('========================= Starting the loop =========================')
    for j in indChosen:
        outputFileName = projRoot + 'v/R/%s/R17fsw%.2ffsh%.2f/%08d.mat' % \
                         (dataset, fScaleWidth, fScaleHeight, j)
        if os.path.isfile(outputFileName) or flagSplit[j] == 0:
            if j % 10000 == 0:
                print('Skipping %d' % j)
            continue
        print('Processing R1 valid triangle selection for %s: %d (j1 = %d, j2 = %d)' %
              (dataset, j, j1, j2))
        houseID0 = int(houseIDList[j])
        viewID0 = int(viewIDList[j])
        scanID0 = scanIDList_house[houseID0]
        cam0 = cam[j, :, :]
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]
        camInv0 = np.linalg.inv(cam0)
        tmp0 = _call_cache_scannet_house_vert_world_0(
            houseID0=houseID0, meta=meta, scannetScanID=scanID0, projRoot=projRoot,
        )
        vertWorld0 = tmp0['vertWorld0']
        face0 = tmp0['face0']
        faceCentroidWorld0 = tmp0['faceCentroidWorld0']
        faceNormalWorld0 = tmp0['faceNormalWorld0']
        faceNyu40ID0 = tmp0['faceNyu40ID0']
        vertCam0 = np.matmul(vertWorld0, camR0.transpose()) + camT0[None, :]
        faceCentroidCam0 = np.matmul(faceCentroidWorld0, camR0.transpose()) + camT0[None, :]
        faceNormalCam0 = np.matmul(faceNormalWorld0, camR0.transpose())

        faceCentroidCamPersp0 = np.stack([
            fScaleWidth * faceCentroidCam0[:, 0] /
                np.clip(faceCentroidCam0[:, 2], a_min=zNear, a_max=np.inf),
            fScaleHeight * faceCentroidCam0[:, 1] /
                np.clip(faceCentroidCam0[:, 2], a_min=zNear, a_max=np.inf),
            faceCentroidCam0[:, 2],
        ], 1)

        # depth operation
        depth0 = imread(
            cache_dataset_root + fileList_house[houseID0] +
            'depth/%d.png' % viewID0)  # , required_dtype='uint16', required_nc=1)
        assert len(depth0.shape) == 2
        assert 'uint16' in str(depth0.dtype)
        assert depth0.max() == 0 or depth0.max() > 300  # uint16
        assert depth0.min() >= 0
        maskUnknown0 = (depth0 == 0)
        depth0 = depth0.astype(np.float32) / 1000.
        depth0[maskUnknown0] = np.nan
        predeterminedResizingScaleOnWidth = focalLengthWidth / fxywxyDepth[j, 0]
        predeterminedResizingScaleOnHeight = focalLengthHeight / fxywxyDepth[j, 1]
        cropDepth0 = croppingCxcyrxry0(
            depth0, [
                ppxyDepth[j, 0],
                ppxyDepth[j, 1],
                winWidth / predeterminedResizingScaleOnWidth / 2.,
                winHeight / predeterminedResizingScaleOnHeight / 2.,
            ],
            padConst=np.nan,
        )
        assert np.nanmin(cropDepth0) >= 0
        depthForUse0 = cv2.resize(
            cropDepth0, (winWidth, winHeight), interpolation=cv2.INTER_LINEAR,
        )
        tmp = depthMap2mesh(depthForUse0, fScaleWidth, fScaleHeight, cutLinkThre=None)
        depthVertCam0 = tmp['vertCam0']
        depthVertWorld0 = np.matmul(depthVertCam0 - camT0[None, :], camR0)  # camR0: inv of transpose
        minWorld0 = depthVertWorld0.min(0)
        maxWorld0 = depthVertWorld0.max(0)
        # minCam0 = depthVertCam0.min(0)
        # maxCam0 = depthVertCam0.max(0)
        slackedMinWorld0 = slacked * minWorld0 + (1 - slacked) * maxWorld0
        slackedMaxWorld0 = (1 - slacked) * minWorld0 + slacked * maxWorld0

        # faceFlag:
        # Do not enforce camcuboid (derived from depthMap) or depthMax (they are the same things)
        # Only enforce worldcuboid (derived from depthMap) (with the -z direction as the gravity)
        faceFlag0 = (
            (np.abs(faceCentroidCamPersp0[:, 0]) < slacked) &
            (np.abs(faceCentroidCamPersp0[:, 1]) < slacked) &
            (faceCentroidWorld0[:, 0] > slackedMinWorld0[0]) &
            (faceCentroidWorld0[:, 1] > slackedMinWorld0[1]) &
            (faceCentroidWorld0[:, 2] > slackedMinWorld0[2]) &
            (faceCentroidWorld0[:, 0] < slackedMaxWorld0[0]) &
            (faceCentroidWorld0[:, 1] < slackedMaxWorld0[1]) &
            (faceCentroidWorld0[:, 2] < slackedMaxWorld0[2]) &
            (faceCentroidCamPersp0[:, 2] > zNear)
        )
        packedFaceFlag0 = np.packbits(faceFlag0, bitorder='big')
        pSaveMat(outputFileName, {
            'packedFaceFlag0': packedFaceFlag0,
        })

        if j in indVisChosen:
            # faceRgb0 = np.ones((face0.shape[0], 3), dtype=np.float32)
            cmap = get_cmap('inferno')
            if False:  # faceNyu40ID0 is not None:
                faceRgb0 = cmap(faceNyu40ID0.astype(np.float32) / 40)[:, :3]
            else:
                faceRgb0 = np.ones((face0.shape[0], 3), dtype=np.float32)
            faceRgb0[faceFlag0, :2] *= 0.6
            faceRgb0[faceFlag0 == 0, -2:] *= 0.6

            assert os.path.isdir(projRoot + 'cache/')
            os.makedirs(visualDir, exist_ok=True)
            dumpPly2(
                visualDir + '%s_%s.ply' % (dataset, j),
                vertCam0, face0, faceRgb0,
            )

            # cuboidWorldVertWorld0, cuboidWorldFace0 = create_cuboid_mesh(minWorld0, maxWorld0)
            cuboidWorldVertWorld0, cuboidWorldFace0 = create_cuboid_mesh(
                slackedMinWorld0, slackedMaxWorld0)
            cuboidWorldFaceRgb0 = np.ones((cuboidWorldFace0.shape[0], 3), dtype=np.float32)
            cuboidWorldFaceRgb0[:, 0] = 0.5
            cuboidWorldFaceRgb0[:, 1] = 0.3
            cuboidWorldFaceRgb0[:, 2] = 0.5
            tmpVertWorld0, tmpFace0, tmpFaceRgb0 = combineMultiShapes_withVertRgb(
                [(vertWorld0, face0, faceRgb0),
                 (cuboidWorldVertWorld0, cuboidWorldFace0, cuboidWorldFaceRgb0)]
            )  # the same with _withFaceRgb
            tmpVertCam0 = np.matmul(tmpVertWorld0, camR0.transpose()) + camT0[None, :]

            dumpPly2(
                visualDir + '%s_%s_with_cuboid.ply' % (dataset, j),
                tmpVertCam0, tmpFace0, tmpFaceRgb0,
            )


if __name__ == '__main__':
    main()
