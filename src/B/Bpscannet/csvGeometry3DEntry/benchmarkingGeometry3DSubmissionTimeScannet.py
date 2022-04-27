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


bt = lambda s: s[0].upper() + s[1:]


def benchmarkingSubmissionTimeScannetSingleViewReconstructionFunc(bsv0, **kwargs):
    scannetMeshCache = kwargs['scannetMeshCache']
    datasets = kwargs['datasets']

    # drawing
    ifRequiresDrawing = kwargs['ifRequiresDrawing']
    pyrenderManager = kwargs['pyrenderManager']

    # predicting
    ifRequiresPredictingHere = kwargs['ifRequiresPredictingHere']

    # Related to benchmarking scores
    voxSize = kwargs['voxSize']
    numMeshSamplingPoint = kwargs['numMeshSamplingPoint']
    cudaDevice = kwargs['cudaDevice']

    tightened = 1. - 4. / voxSize
    prDistThre = 0.05

    # from bsv0
    index = int(bsv0['index'])
    did = int(bsv0['did'])
    houseID = int(bsv0['houseID'])
    viewID = int(bsv0['viewID'])
    datasetObj = datasets[did]
    dataset = datasetObj.datasetConf['dataset']
    assert dataset.startswith('scannetGivenRender')
    datasetConf = datasetObj.datasetConf

    # cam
    cam0 = bsv0['cam']
    camR0 = cam0[:3, :3]
    camT0 = cam0[:3, 3]
    camInv0 = np.linalg.inv(cam0)
    bsv0['camInv'] = camInv0

    # fxywxy
    bsv0['focalLengthWidth'] = datasetConf['focalLengthWidth']
    bsv0['focalLengthHeight'] = datasetConf['focalLengthHeight']
    bsv0['winWidth'] = datasetConf['winWidth']
    bsv0['winHeight'] = datasetConf['winHeight']
    bsv0['fxywxy'] = np.array([
        bsv0['focalLengthWidth'], bsv0['focalLengthHeight'],
        bsv0['winWidth'], bsv0['winHeight'],
    ], dtype=np.float32)
    bsv0['fScaleWidth'] = datasetConf['fScaleWidth']
    bsv0['fScaleHeight'] = datasetConf['fScaleHeight']
    bsv0['zNear'] = datasetConf['zNear']

    # re-load the mesh and produce the mesh viewport-only
    tmp0 = scannetMeshCache.call_cache_scannet_house_vert_world_0(
        houseID0=houseID,
        scannetFile=datasetObj.fileList_house[houseID],
        scannetScanID=datasetObj.scanIDList_house[houseID],
        verbose=True,
        original_dataset_root=datasetObj.original_dataset_root,
    )
    fullVertWorld0 = tmp0['vertWorld0']
    fullFace0 = tmp0['face0']
    bsv0['fullVertWorld'] = fullVertWorld0
    bsv0['fullFace'] = fullFace0
    packedFaceFlag0 = udl('mats_R1fsw%.2ffwh%.2f_packedFaceFlag0' %
                          (datasetConf['fScaleWidth'],
                           datasetConf['fScaleHeight']),
                          dataset, index)
    faceFlag0 = np.unpackbits(packedFaceFlag0, bitorder='big')[:fullFace0.shape[0]].astype(bool)
    viewVertWorld0 = fullVertWorld0.copy()
    viewFace0 = fullFace0[faceFlag0]
    viewVertWorld0, viewFace0 = trimVert(viewVertWorld0, viewFace0)
    bsv0['viewVertWorld'] = viewVertWorld0
    bsv0['viewFace'] = viewFace0
    bsv0['boundMinWorld'] = viewVertWorld0.min(0)
    bsv0['boundMaxWorld'] = viewVertWorld0.max(0)
    bsv0['fullFaceRgb'] = np.ones_like(fullFace0)
    bsv0['fullFaceRgb'][faceFlag0, 2] = 0.6
    bsv0['fullFaceRgb'][faceFlag0 == 0, 0] = 0.6
    del tmp0

    # world2cam
    for meshName in ['view', 'full']:
        bsv0['%sVertCam' % meshName] = np.matmul(
            bsv0['%sVertWorld' % meshName], camR0.transpose()) + camT0[None, :]
    bsv0['boundMinCam'] = bsv0['viewVertCam'].min(0)
    bsv0['boundMaxCam'] = bsv0['viewVertCam'].max(0)
    boundMinWorld0, boundMaxWorld0 = bsv0['viewVertWorld'].min(0), bsv0['viewVertWorld'].max(0)
    s = 0.02
    boundMinWorld0, boundMaxWorld0 = (1 + s) * boundMinWorld0 - s * boundMaxWorld0, \
        (1 + s) * boundMaxWorld0 - s * boundMinWorld0
    bsv0['boundMinWorld'], bsv0['boundMaxWorld'] = boundMinWorld0, boundMaxWorld0

    if ifRequiresPredictingHere:
        # return: predVertCam, predFace

        # During monitoring
        # If you want to evaluate both the predVertCam and the depthPredVertCam
        # You need to go through this function twice,
        # And each time, provide a different doPred0Func
        # in which one returns the predVertCam, and the other returns the depthPredVertCam

        # bsv0 = doPred0Func(
        #     bsv0, voxSize=voxSize, models=models, cudaDevice=cudaDevice,
        #     verboseBatchForwarding=verboseBatchForwarding, Trainer=Trainer,
        # )
        bsv0 = kwargs['doPred0Func'](bsv0, **kwargs)
    else:
        if 'predVertWorld' in bsv0.keys() and 'predVertCam' not in bsv0.keys():
            bsv0['predVertCam'] = np.matmul(
                bsv0['predVertWorld'], camR0.transpose()
            ) + camT0[None, :]

    # trim away frustum border mesh triangles for the purpose of evaluation
    # trim away outside-room border mesh triangles for the purpose of evaluation
    for meshName in ['pred', 'view']:
        vertCam0 = bsv0['%sVertCam' % meshName].copy()
        face0 = bsv0['%sFace' % meshName].copy()
        vertWorld0 = np.matmul(vertCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse
        if face0.shape[0] == 0:
            vertCam0 = np.eye(3).astype(np.float32)
            face0 = np.array([[0, 1, 2]], dtype=np.int32)
        vertCamPersp0 = np.stack([
            bsv0['fScaleWidth'] * vertCam0[:, 0] /
            np.clip(vertCam0[:, 2], a_min=bsv0['zNear'], a_max=np.inf),
            bsv0['fScaleHeight'] * vertCam0[:, 1] /
            np.clip(vertCam0[:, 2], a_min=bsv0['zNear'], a_max=np.inf),
            vertCam0[:, 2],
        ], 1)
        faceVertCamPersp0 = vertInfo2faceVertInfoNP(
            vertCamPersp0[None], face0[None])[0]
        faceExtremeCamPersp0 = np.abs(faceVertCamPersp0).max(1)
        faceVertWorld0 = vertInfo2faceVertInfoNP(
            vertWorld0[None], face0[None])[0]
        faceMaxWorld0 = faceVertWorld0.max(1)
        faceMinWorld0 = faceVertWorld0.min(1)
        flagIn = (
                (np.abs(faceExtremeCamPersp0[:, 0]) < tightened) &
                (np.abs(faceExtremeCamPersp0[:, 1]) < tightened)
        )
        for xyz in range(3):
            flagIn &= (
                          (faceMaxWorld0[:, xyz] < bsv0['boundMaxWorld'][xyz]) &
                          (faceMinWorld0[:, xyz] > bsv0['boundMinWorld'][xyz])
            )
        if flagIn.sum() == 0:
            flagIn[0] = True
        face0 = face0[flagIn]
        vertCam0, face0 = trimVert(vertCam0, face0)
        bsv0['eval%sVertCam' % bt(meshName)] = vertCam0
        bsv0['eval%sFace' % bt(meshName)] = face0

    # cam2world
    for meshName in ['evalPred', 'evalView']:
        bsv0['%sVertWorld' % meshName] = np.matmul(
            bsv0['%sVertCam' % meshName] - camT0[None, :], camR0  # transpose of inverse
        )

    bsv0['evalView2VertWorld'] = bsv0['evalViewVertWorld'].copy()
    bsv0['evalView2Face'] = bsv0['evalViewFace'].copy()
    bsv0['evalView2FaceRgb'] = 0.6 * np.ones((bsv0['evalView2Face'].shape[0], 3), dtype=np.float32)

    # get face centroid
    #   naming convention: vXxxWorld, fcXxxWorld are vert/faceCentroid PC of mesh xxx
    #   without prefix: it is actually the standard numPoint sampling result
    for meshName in ['evalPred', 'evalView']:
        sysLabel = 'world'
        faceCentroid0 = vertInfo2faceVertInfoNP(
            bsv0['%sVert%s' % (meshName, bt(sysLabel))][None],
            bsv0['%sFace' % meshName][None])[0].mean(1)
        bsv0['fc%sPCxyz%s' % (bt(meshName), bt(sysLabel))] = faceCentroid0
        bsv0['v%sPCxyz%s' % (bt(meshName), bt(sysLabel))] = \
            bsv0['%sVert%s' % (meshName, bt(sysLabel))].copy()

    # pc sampling, for benchmarking and visualization
    for meshName in ['full', 'evalView', 'evalPred', 'evalView2']:
        pcName = meshName
        sysLabel = 'world'
        vert0_thgpu = torch.from_numpy(bsv0['%sVert%s' % (meshName, bt(sysLabel))]).to(cudaDevice)
        face0_thgpu = torch.from_numpy(bsv0['%sFace' % meshName]).long().to(cudaDevice)
        faceVert0_thgpu = vertInfo2faceVertInfoTHGPU(vert0_thgpu[None], face0_thgpu[None])[0]
        faceRawNormal0_thgpu = torch.cross(faceVert0_thgpu[:, 1, :] - faceVert0_thgpu[:, 0, :],
                                           faceVert0_thgpu[:, 2, :] - faceVert0_thgpu[:, 0, :])
        faceArea0_thgpu = torch.norm(faceRawNormal0_thgpu, dim=1, p=2)
        faceNormal0_thgpu = torch.div(
            faceRawNormal0_thgpu, torch.clamp(faceArea0_thgpu, datasetConf['zNear'])[:, None])
        faceCumsumWeight0_thgpu = torch.cumsum(faceArea0_thgpu, dim=0)
        point0_thgpu, _, _, _ = mesh_weighted_sampling_given_normal(
            numMeshSamplingPoint, faceCumsumWeight0_thgpu, faceVert0_thgpu,
            faceNormal0_thgpu, {}, {})
        bsv0['%sPCxyz%s' % (pcName, bt(sysLabel))] = point0_thgpu.detach().cpu().numpy()

    # depth to point cloud
    for depthMapName in ['depthForUse', 'depthPred']:
        if depthMapName in bsv0.keys():
            if depthMapName == 'depthPred':
                bsv0[depthMapName][bsv0[depthMapName] < 0] = bsv0['zNear']
            bsv0['%sPCxyzCam' % depthMapName], _ = directDepthMap2PCNP(
                bsv0[depthMapName][0], bsv0['fScaleWidth'], bsv0['fScaleHeight']
            )
            bsv0['%sPCxyzWorld' % depthMapName] = np.matmul(
                bsv0['%sPCxyzCam' % depthMapName] - camT0[None, :],
                camR0  # transpose of inverse
            )

    # predicted rendered depth
    sysLabel = 'world'
    meshName = 'evalPred'
    camInv0 = bsv0['camInv']
    fxywxy0 = bsv0['fxywxy']
    vert0 = bsv0['%sVert%s' % (meshName, bt(sysLabel))]
    face0 = bsv0['%sFace' % meshName]
    pyrenderManager.clear()
    pyrenderManager.add_camera(fxywxy0, camInv0)
    pyrenderManager.add_plain_mesh(vert0, face0)
    tmp = pyrenderManager.render()
    d = tmp[1]
    d[(d <= 0) | (d >= 10) | (np.isfinite(d) == 0)] = np.nan
    bsv0['depthRenderedFromEvalPred2'] = d

    # polyfit
    bsv0['depthForUse2'] = bsv0['depthForUse'][0, :, :]
    a, b, _ = affinePolyfitWithNaN(bsv0['depthRenderedFromEvalPred2'], bsv0['depthForUse2'])
    bsv0['affinePolyfitA'] = float(a)
    bsv0['affinePolyfitB'] = float(b)
    del a, b

    # benchmarking - 3D
    for pcName in ['evalPred']:
        pcName0 = pcName
        pcName1 = 'evalView'
        sysLabel = 'world'
        if '%sPCxyz%s' % (pcName0, bt(sysLabel)) in bsv0.keys():
            tmp0 = packageCDF1(
                bsv0['%sPCxyz%s' % (pcName0, bt(sysLabel))],
                bsv0['%sPCxyz%s' % (pcName1, bt(sysLabel))],
                prDistThre=prDistThre, cudaDevice=cudaDevice,
            )
            for k in ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']:
                bsv0['%sPCBenMetric%s' % (pcName0, bt(k))] = tmp0[k]

            # 3D fit
            xyzWorld0 = bsv0['%sPCxyzWorld' % pcName0]
            cam0 = bsv0['cam']
            camR0 = cam0[:3, :3]
            camT0 = cam0[:3, 3]
            xyzCam0 = np.matmul(xyzWorld0, camR0.transpose()) + camT0[None, :]
            fScaleWidth0, fScaleHeight0 = bsv0['fScaleWidth'], bsv0['fScaleHeight']
            zNear = bsv0['zNear']
            xyzCamPersp0 = np.stack([
                xyzCam0[:, 0] * fScaleWidth0 / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
                xyzCam0[:, 1] * fScaleHeight0 / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
                xyzCam0[:, 2],
            ], 1)
            a, b = bsv0['affinePolyfitA'], bsv0['affinePolyfitB']
            xyzCamPersp0[:, 2] = a * xyzCamPersp0[:, 2] + b
            xyzCam0 = np.stack([
                xyzCamPersp0[:, 0] * xyzCamPersp0[:, 2] / fScaleWidth0,
                xyzCamPersp0[:, 1] * xyzCamPersp0[:, 2] / fScaleHeight0,
                xyzCamPersp0[:, 2]
            ], 1)
            xyzWorld0 = np.matmul(xyzCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse
            tmp1 = packageCDF1(
                xyzWorld0,
                bsv0['%sPCxyz%s' % (pcName1, bt(sysLabel))],
                prDistThre=prDistThre, cudaDevice=cudaDevice,
            )
            for k in ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']:
                bsv0['%sPCBenFit%s' % (pcName0, bt(k))] = tmp1[k]

            del tmp0, tmp1, xyzWorld0, xyzCam0, xyzCamPersp0, cam0, camR0, camT0
            del fScaleWidth0, fScaleHeight0, zNear, a, b

    # coloring
    tmp = packageCDF1(
        abTheMeshInTheWorldSys(
            bsv0['vEvalPredPCxyzWorld'], bsv0['cam'][:3, :3], bsv0['cam'][:3, 3],
            bsv0['fScaleWidth'], bsv0['fScaleHeight'],
            bsv0['affinePolyfitA'], bsv0['affinePolyfitB'], bsv0['zNear'],
        ),
        bsv0['vEvalViewPCxyzWorld'],
        prDistThre=prDistThre, cudaDevice=cudaDevice,
    )
    bsv0['evalPredVertRgb'] = 0.2 * np.ones(
        (bsv0['evalPredVert%s' % bt(sysLabel)].shape[0], 3), dtype=np.float32)
    bsv0['evalPredVertRgb'][tmp['distP'] < prDistThre, 2] = 0.6  # light blue precision correct
    bsv0['evalPredVertRgb'][tmp['distP'] >= prDistThre, 0] = 0.6  # red precision wrong
    bsv0['evalViewVertRgb'] = 0.07 * np.ones(
        (bsv0['evalViewVert%s' % bt(sysLabel)].shape[0], 3), dtype=np.float32)
    bsv0['evalViewVertRgb'][tmp['distR'] < prDistThre, 2] = 0.4  # navy recall correct
    bsv0['evalViewVertRgb'][tmp['distR'] >= prDistThre, :2] = 0.6  # yellow recall wrong

    tmp = packageCDF1(
        abTheMeshInTheWorldSys(
            bsv0['fcEvalPredPCxyzWorld'], bsv0['cam'][:3, :3], bsv0['cam'][:3, 3],
            bsv0['fScaleWidth'], bsv0['fScaleHeight'],
            bsv0['affinePolyfitA'], bsv0['affinePolyfitB'], bsv0['zNear'],
        ),
        bsv0['fcEvalViewPCxyzWorld'],
        prDistThre=prDistThre, cudaDevice=cudaDevice,
    )
    bsv0['evalPredFaceRgb'] = 0.2 * np.ones(
        (bsv0['evalPredFace'].shape[0], 3), dtype=np.float32)
    bsv0['evalPredFaceRgb'][tmp['distP'] < prDistThre, 2] = 0.6  # light blue precision correct
    bsv0['evalPredFaceRgb'][tmp['distP'] >= prDistThre, 0] = 0.6  # red precision wrong
    bsv0['evalViewFaceRgb'] = 0.07 * np.ones(
        (bsv0['evalViewFace'].shape[0], 3), dtype=np.float32)
    bsv0['evalViewFaceRgb'][tmp['distR'] < prDistThre, 2] = 0.4  # navy recall correct
    bsv0['evalViewFaceRgb'][tmp['distR'] >= prDistThre, :2] = 0.6  # yellow recall wrong

    # benchmarking - 2.5D (direct depth prediction)
    for key in ['depthRenderedFromEvalPred2']:
        depthKey0 = key
        depthKey1 = 'depthForUse2'
        if depthKey0 in bsv0.keys():
            tmp = packageDepth(bsv0[depthKey0], bsv0[depthKey1])
            for k in ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE',
                      'r1', 'r2', 'r3', 'complete']:
                bsv0['%sDBenMetric%s' % (depthKey0, bt(k))] = tmp[k]
            a = bsv0['affinePolyfitA']
            b = bsv0['affinePolyfitB']
            tmp = packageDepth(bsv0[depthKey0] * a + b, bsv0[depthKey1])
            for k in ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE',
                      'r1', 'r2', 'r3', 'complete']:
                bsv0['%sDBenFit%s' % (depthKey0, bt(k))] = tmp[k]

    # draw the mesh
    # (even if not drawing required, you still need to render once for depthRendered)
    if ifRequiresDrawing:
        for meshName in ['evalView2', 'evalView', 'evalPred']:
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            prefix = '%s%s%sMeshDrawingPackage' % (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))

            numView = 9
            ungrav0 = np.array([0, 0, 1], dtype=np.float32)
            EPick0List, LCenter0 = pickInitialObservationPoints(
                bsv0['EWorld'], bsv0['LWorld'], ungrav0,
                bsv0['depthMax'], numView=9,
            )
            EPick0List = EPick0List[[0, 1, 8], :]
            EPick0ListElevated = EPick0List + 0.7 * ungrav0[None, :]
            EPick0ListDowned = EPick0List - 0.5 * ungrav0[None, :]
            EPick0List = np.concatenate([EPick0List, EPick0ListElevated, EPick0ListDowned], 0)
            LCenter0 = 1.5 * LCenter0 - 0.5 * bsv0['EWorld']
            bsv0[prefix + 'NumView'] = numView
            bsv0[prefix + 'EPick0List'] = EPick0List
            bsv0[prefix + 'LCenter0'] = LCenter0
            bsv0[prefix + 'UPick0List'] = np.tile(ungrav0[None, :], (numView, 1))
            bsv0[prefix + 'UPick0List'][0, :] = bsv0['UWorld']
            bsv0[prefix + 'ViewColor'] = np.ones(
                (numView, bsv0['winHeight'], bsv0['winWidth'], 3), dtype=np.float32)
            for v in range(numView):
                pyrenderManager.clear()
                pyrenderManager.add_point_light(
                    pointLoc=1.2 * EPick0List[v] - 0.2 * LCenter0, intensity=0.2,
                    color=[200., 200., 200.]
                )
                viewCam0 = ELU02cam0(np.concatenate(
                    [EPick0List[v], LCenter0, bsv0[prefix + 'UPick0List'][v]], 0))
                viewCamInv0 = np.linalg.inv(viewCam0)
                pyrenderManager.add_camera(bsv0['fxywxy'], viewCamInv0)
                pyrenderManager.add_vertRgb_mesh_via_faceRgb(
                    bsv0['%sVertWorld' % meshName], bsv0['%sFace' % meshName],
                    bsv0['%sFaceRgb' % meshName])
                tmp = pyrenderManager.render()
                bsv0[prefix + 'ViewColor'][v, :, :, :] = tmp[0].astype(np.float32) / 255.
                if v == 0:
                    d = tmp[1]
                    d[(d <= 0) | (d >= 10) | (np.isfinite(d) == 0)] = np.nan
                    bsv0['depthRenderedFromEvalPred2'] = d
            # draw floor plan here
            floorAxis0, floorAxis1 = 0, 1  # world sys floor plan

            def f(ax):
                for mn in ['evalView', meshName]:
                    pp = bsv0['%sPCxyz%s' % (mn, bt(sysLabel))]
                    ax.scatter(
                        pp[:, floorAxis0], pp[:, floorAxis1],
                        c='r' if mn == 'evalPred' else 'b',
                        s=0.01 if mn == 'evalPred' else 0.003, marker='.')
                e = bsv0[prefix + 'EPick0List']
                l = bsv0[prefix + 'LCenter0']
                for v in range(numView):
                    ax.scatter(e[v, floorAxis0], e[v, floorAxis1], c='k', s=20, marker='o')
                ax.scatter(l[floorAxis0], l[floorAxis1], c='k', s=50, marker='x')
                ax.scatter(e[0, floorAxis0], e[0, floorAxis1], c='k', s=50, marker='o')

            bsv0[prefix + 'FloorPlan'] = getPltDraw(f)
    else:
        pass

    return bsv0


def abTheMeshInTheWorldSys(xyzWorld0, camR0, camT0, fScaleWidth, fScaleHeight, a, b, zNear):
    assert len(xyzWorld0.shape) == 2 and xyzWorld0.shape[1] == 3
    assert camR0.shape == (3, 3)
    assert camT0.shape == (3, )
    assert type(fScaleWidth) is float
    assert type(fScaleHeight) is float
    assert type(a) is float
    assert type(b) is float
    assert type(zNear) is float
    xyzCam0 = np.matmul(xyzWorld0, camR0.transpose()) + camT0[None, :]
    xyzCamPersp0 = np.stack([
        xyzCam0[:, 0] * fScaleWidth / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 1] * fScaleHeight / np.clip(xyzCam0[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam0[:, 2],
    ], 1)
    xyzCamPersp0[:, 2] = a * xyzCamPersp0[:, 2] + b
    xyzCam0_new = np.stack([
        xyzCamPersp0[:, 0] * xyzCamPersp0[:, 2] / fScaleWidth,
        xyzCamPersp0[:, 1] * xyzCamPersp0[:, 2] / fScaleHeight,
        xyzCamPersp0[:, 2]
    ], 1)
    xyzWorld0_new = np.matmul(xyzCam0_new - camT0[None, :], camR0)  # camR0: transpose of inverse
    return xyzWorld0_new
