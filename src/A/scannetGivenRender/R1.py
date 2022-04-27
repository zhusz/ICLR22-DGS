# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# bash for parallel : for g in {0..26}; do J1=$((g*100000)) J2=$(((g+1)*100000)) oap R1.py & done
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from collections import OrderedDict
from codes_py.np_ext.mat_io_v1 import pLoadMat
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from codes_py.np_ext.np_image_io_v1 import imread
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.np_ext.mat_io_v1 import pSaveMat
import trimesh
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
    fileID = kwargs['fileID']
    assert type(houseID0) is int
    original_dataset_root = projRoot + 'remote_fastdata/scannet/'
    if houseID0 not in meta['houseVertWorldCache']['scannet'].keys():
        print('    [CacheScannet] Loading house vert world for scannet - %d' % houseID0)
        t = time.time()
        # mat = pLoadMat(projRoot + 'remote_fastdata/cache/scannet/detailedMesh/%s.mat' % scannetScanID)
        mesh = trimesh.load(original_dataset_root + fileID + '%s_vh_clean_2.ply' % scannetScanID,
                            process=True)
        vert0 = np.array(mesh.vertices, dtype=np.float32)
        face0 = np.array(mesh.faces, dtype=np.int32)
        faceVert0 = vertInfo2faceVertInfoNP(
            vert0[None], face0[None],
        )[0]
        faceCentroid0 = faceVert0.mean(1)
        faceNormal0 = np.cross(
            faceVert0[:, 1, :] - faceVert0[:, 0, :],
            faceVert0[:, 2, :] - faceVert0[:, 0, :],
        )
        faceNormal0 = np.divide(
            faceNormal0,
            np.linalg.norm(faceNormal0, ord=2, axis=1)[:, None] + 1.e-6,
        )
        print('Elapsed Time is: %.5f seconds.' % (time.time() - t))
        meta['houseVertWorldCache']['scannet'][houseID0] = {
            'vertWorld0': vert0,
            'face0': face0,
            'faceCentroidWorld0': faceCentroid0,
            'faceNormalWorld0': faceNormal0,
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
    visualDir = projRoot + 'cache/dataset/scannetGivenRender/R1/'
    assert os.path.isdir(projRoot + 'cache/')
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 100, 'R1')

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
    A1_house = udl('pkl_A1_', dataset_house)
    scanIDList_house = A1_house['scanIDList']
    fileList_house = A1_house['fileList']

    cam = ELU2cam(np.concatenate([E, L, U], 1))
    fScaleWidth = 2. * focalLengthWidth / winWidth
    fScaleHeight = 2. * focalLengthHeight / winHeight
    # indVisChosen = range(5)
    indVisChosen = indChosen
    meta = {'houseVertWorldCache': {'scannet': {}}}
    zNear = 1.e-6
    slacked = 1.05
    for j in indChosen:
        print('Processing R1 valid triangle selection for %s: %d (j1 = %d, j2 = %d)' %
              (dataset, j, j1, j2))
        houseID0 = int(houseIDList[j])
        viewID0 = int(viewIDList[j])
        scanID0 = scanIDList_house[houseID0]
        fileID0 = fileList_house[houseID0]
        cam0 = cam[j, :, :]
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]
        camInv0 = np.linalg.inv(cam0)
        tmp0 = _call_cache_scannet_house_vert_world_0(
            houseID0=houseID0, meta=meta, scannetScanID=scanID0, fileID=fileID0, projRoot=projRoot,
        )
        vertWorld0 = tmp0['vertWorld0']
        face0 = tmp0['face0']
        faceCentroidWorld0 = tmp0['faceCentroidWorld0']
        faceNormalWorld0 = tmp0['faceNormalWorld0']
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

        faceFlag0 = (
            (np.abs(faceCentroidCamPersp0[:, 0]) < slacked) &
            (np.abs(faceCentroidCamPersp0[:, 1]) < slacked) &
            (faceCentroidCamPersp0[:, 2] > zNear)
        )
        packedFaceFlag0 = np.packbits(faceFlag0, bitorder='big')
        # pSaveMat(projRoot + 'v/R/%s/R1fsw%.2ffsh%.2f/%08d.mat' %
        #          (dataset, focalLengthWidth, focalLengthHeight, j), {
        #     'packedFaceFlag0': packedFaceFlag0,
        # })

        if j in indVisChosen:
            # put into html to visualize (colored mesh for showing valid / invalid triangles)
            # 1. crop
            img0 = imread(cache_dataset_root + fileList_house[houseID0] + 'color/%d.jpg' % viewID0)
            # your resizing scaling ratio is pre-determined here.
            predeterminedResizingScaleOnWidth = focalLengthWidth / fxywxyColor[j, 0]
            predeterminedResizingScaleOnHeight = focalLengthHeight / fxywxyColor[j, 1]
            cropImg0 = croppingCxcyrxry0(
                img0, [
                    ppxyColor[j, 0],
                    ppxyColor[j, 1],
                    winWidth / predeterminedResizingScaleOnWidth / 2.,
                    winHeight / predeterminedResizingScaleOnHeight / 2.,
                ]
            )
            del predeterminedResizingScaleOnWidth, predeterminedResizingScaleOnHeight
            returnedImg0 = cv2.resize(
                cropImg0, (winWidth, winHeight),
                interpolation=cv2.INTER_CUBIC,
            )
            depth0 = imread(cache_dataset_root + fileList_house[houseID0] + 'depth/%d.png' % viewID0,
                            required_dtype='uint16', required_nc=1)
            depth0 = depth0.astype(np.float32) / 1000.
            predeterminedResizingScaleOnWidth = focalLengthWidth / fxywxyDepth[j, 0]
            predeterminedResizingScaleOnHeight = focalLengthHeight / fxywxyDepth[j, 1]
            cropDepth0 = croppingCxcyrxry0(
                depth0, [
                    ppxyDepth[j, 0],
                    ppxyDepth[j, 1],
                    winWidth / predeterminedResizingScaleOnWidth / 2.,
                    winHeight / predeterminedResizingScaleOnHeight / 2.,
                ]
            )
            returnedDepth0 = cv2.resize(
                cropDepth0, (winWidth, winHeight),
                interpolation=cv2.INTER_CUBIC,
            )
            del predeterminedResizingScaleOnWidth, predeterminedResizingScaleOnHeight

            # 2. render
            renderer = pyrender.OffscreenRenderer(winWidth, winHeight)
            scene = pyrender.Scene()
            renderer.viewport_width = winWidth
            renderer.viewport_heigth = winHeight
            scene.clear()
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
                vertices=vertWorld0, faces=face0,
            )))
            camObj = pyrender.IntrinsicsCamera(
                cx=winWidth / 2, cy=winHeight / 2,
                fx=focalLengthWidth, fy=focalLengthHeight,
            )
            scene.add(camObj, pose=fix_pose(camInv0))
            pl = pyrender.PointLight(color=[50., 50., 200.], intensity=0.8)
            light_matrix = np.eye(4).astype(np.float32)
            light_matrix[:3, 3] = np.array([2, 2, 2], dtype=np.float32)
            scene.add(pl, pose=fix_pose(light_matrix))
            tmp = renderer.render(scene)
            pyrenderColor0 = tmp[0].astype(np.float32) / 255.
            pyrenderDepth0 = tmp[1]

            # 3. put into html
            summary0 = OrderedDict([])
            brInds = [0, ]
            summary0['originalImg0'] = img0
            summary0['returnedImg0'] = returnedImg0
            summary0['pyrenderImg0'] = pyrenderColor0
            def f(ax):
                ax.scatter(faceCentroidWorld0[faceFlag0, 0], faceCentroidWorld0[faceFlag0, 1],
                           c='b', s=0.02, marker='.')
                ax.scatter(faceCentroidWorld0[faceFlag0 == 0, 0], faceCentroidWorld0[faceFlag0 == 0, 1],
                           c='r', s=0.02, marker='.')
                ax.scatter(E[j, 0], E[j, 1], c='k', s=20, marker='o')
                ax.scatter(L[j, 0], L[j, 1], c='m', s=20, marker='x')
            summary0['floorPlanWorld'] = getPltDraw(f)
            def g(ax):
                ax.scatter(faceCentroidCam0[faceFlag0, 0], faceCentroidCam0[faceFlag0, 2],
                           c='b', s=0.02, marker='.')
                ax.scatter(faceCentroidCam0[faceFlag0 == 0, 0], faceCentroidCam0[faceFlag0 == 0, 2],
                           c='r', s=0.02, marker='.')
                ax.scatter(0, 0, c='k', s=20, marker='o')
                ax.scatter(0, 1, c='m', s=20, marker='x')
            summary0['floorPlanCam'] = getPltDraw(g)
            brInds.append(len(summary0))
            summary0['originalDepth0'] = getImshow(depth0)
            summary0['returnedDepth0'] = getImshow(returnedDepth0)
            summary0['pyrenderDepth0'] = getImshow(pyrenderDepth0)
            brInds.append(len(summary0))
            htmlStepper.step2(
                summary0, txt0=None, brInds=brInds,
                headerMessage='Dataset: %s, Index: %s, houseID: %d, houseScanID: %s, viewID: %d' %
                              (dataset, j, houseID0, scanID0, viewID0),
                subMessage='',
            )

            # 4. dump as ply
            faceRgb0 = np.ones((face0.shape[0], 3), dtype=np.float32)
            faceRgb0[faceFlag0, :2] = 0.6
            faceRgb0[faceFlag0 == 0, -2:] = 0.6
            dumpPly2(
                visualDir + '%s_%s.ply' % (dataset, j),
                vertCam0, face0, faceRgb0,
            )

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
