# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# To separate the metric prediction and the fitted prediction. So easier to debug and visualize what is happening between the metric prediction and the fitted prediction.
# So metric prediction and fitted prediction would have two different sets of coloring routines.
# When delineating the space, metric/fitted now has their own delineated space. 
# When visualizing to the html, we only draw the fitted one. But for dumping the mesh, we dump both.

# Notes: the labeled mesh (evalViewMesh) there is only one. (One flagIn)
# The evalPredMetric and evalPredFit meshes will be two. (Two different flagIn, benchmarkings, coloring)
# All the metrics will be evaluated twice.

from UDLv3 import udl
import torch
from .dumpCsvGeometry3D import abTheMeshInTheWorldSys, abTheMeshInTheCamSys, \
    judgeWhichTrianglesFlagIn, exitWithNanBenchmarking
import numpy as np
import trimesh
import pymesh
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert, trimVertGeneral
from codes_py.toolbox_3D.benchmarking_v1 import packageCDF1, packageDepth, affinePolyfitWithNaN
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP, vertInfo2faceVertInfoTHGPU
from codes_py.toolbox_3D.self_sampling_v1 import mesh_weighted_sampling_given_normal_fixed_rand
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.draw_mesh_v1 import pickInitialObservationPoints, getUPick0List
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, getRotationMatrixBatchNP
from codes_py.toolbox_3D.mesh_subdivision_v1 import selectiveMeshSubdivision
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, to_heatmap
# easy debugging
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2
from skimage.io import imsave
from codes_py.np_ext.mat_io_v1 import pSaveMat
from skimage.io import imread
np.set_printoptions(suppress=True)


bt = lambda s: s[0].upper() + s[1:]


def benchmarkingGeometry3DHmFunc(bsv0, **kwargs):
    datasetObj = kwargs['datasetObj']
    hmMeshCache = kwargs['hmMeshCache']
    raiseOrTrace = kwargs['raiseOrTrace']
    assert datasetObj.datasetConf['dataset'].startswith('hmRenderOm')
    datasetConf = datasetObj.datasetConf

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

    # orinal rgb / depth
    imgRaw_file_name = datasetObj.rendering_rgb_root + '%05d-%s/point_%d_view_%d_domain_rgb.png' % \
        (bsv0['houseID'], bsv0['houseSha'], bsv0['pointID'], bsv0['viewID'])
    imgRaw0 = imread(imgRaw_file_name)
    assert imgRaw0.dtype == np.uint8
    assert imgRaw0.shape == (
        datasetObj.resolution[int(bsv0['index'])], 
        datasetObj.resolution[int(bsv0['index'])],
        3
    )
    assert 1.5 < imgRaw0.max() <= 255.5
    imgRaw0 = imgRaw0.astype(np.float32) / 255.
    bsv0['imgRaw'] = imgRaw0.transpose((2, 0, 1))
    depthRaw_file_name = datasetObj.rendering_depth_zbuffer_root + \
        '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' % \
        (bsv0['houseID'], bsv0['houseSha'], bsv0['pointID'], bsv0['viewID'])
    depthRaw0 = imread(depthRaw_file_name)
    assert depthRaw0.dtype == np.uint16
    assert depthRaw0.shape == (
        datasetObj.resolution[int(bsv0['index'])], 
        datasetObj.resolution[int(bsv0['index'])],
    )
    assert depthRaw0.max() > 300
    assert depthRaw0.min() >= 0
    depthRaw0 = depthRaw0.astype(np.float32) / 512.  # 512 is the magic number for hmRenderOm
    bsv0['depthRaw'] = depthRaw0[None, :, :]
    del imgRaw0, imgRaw_file_name, depthRaw0, depthRaw_file_name

    # cam
    cam0 = bsv0['cam']
    camR0 = cam0[:3, :3]
    camT0 = cam0[:3, 3]
    camInv0 = np.linalg.inv(cam0)
    bsv0['camInv'] = camInv0

    # fxywxy
    assert bsv0['ifNeedPixelAugmentation'] == 0  # benchmarking no augmentation
    assert bsv0['ifNeedCroppingAugmentation'] in [0, 1]  # benchmarking no augmentation
    # If you relaxed this to the fixed fScale setting, you need to redelinate the evaluation space scope
    assert bsv0['ifNeedMirrorAugmentation'] == 0  # benchmarking no augmentation
    bsv0['focalLengthWidth'] = bsv0['focalLengthWidth']
    bsv0['focalLengthHeight'] = bsv0['focalLengthHeight']
    bsv0['winWidth'] = bsv0['winWidth']
    bsv0['winHeight'] = bsv0['winHeight']
    bsv0['fxywxy'] = np.array([
        bsv0['focalLengthWidth'], bsv0['focalLengthHeight'],
        bsv0['winWidth'], bsv0['winHeight'],
    ], dtype=np.float32)
    bsv0['fScaleWidth'] = bsv0['fScaleWidth']
    bsv0['fScaleHeight'] = bsv0['fScaleHeight']
    bsv0['zNear'] = datasetConf['zNear']

    # re-load the mesh and produce the mesh viewport-only
    tmp0 = hmMeshCache.call_cache_hm_house_vert_world_0(
        houseID0=int(bsv0['houseID']), verbose=True,
    )
    fullVertWorld0 = tmp0['vertWorld0']
    fullFace0 = tmp0['face0']
    bsv0['fullVertWorld'] = fullVertWorld0
    bsv0['fullFace'] = fullFace0
    nFace = int(fullFace0.shape[0])
    faceFlag0 = bsv0['faceFlagUntrimmed'][:nFace].astype(bool)
    fullFaceRgb0 = 0.5 * np.ones((fullFace0.shape[0], 3), dtype=np.float32)
    fullFaceRgb0[faceFlag0 == 1, 0] += 0.2
    fullFaceRgb0[faceFlag0 == 0, 2] += 0.2
    bsv0['fullFaceRgb'] = fullFaceRgb0
    bsv0['fullVertRgb'] = trimesh.visual.color.face_to_vertex_color(
        mesh=trimesh.Trimesh(vertices=fullVertWorld0, faces=fullFace0, process=False),
        face_colors=fullFaceRgb0,
        dtype=np.float32
    )[:, :3] / 255.
    viewVertWorld0 = fullVertWorld0.copy()
    viewFace0 = fullFace0[faceFlag0]
    viewVertWorld0, viewFace0 = trimVert(viewVertWorld0, viewFace0)
    viewVertWorld0, viewFace0 = \
        selectiveMeshSubdivision(viewVertWorld0, viewFace0,
                                 order=1, edgeLengthThre=0.1, verbose=1)
    # subseting the current view port (give that the provided fov might be smaller than the original one in R17)
    # No need - viewMesh can keeps to be the original fov, but evalView will make sure only the 
    # augmented cropped part can count
    bsv0['viewVertWorld'] = viewVertWorld0
    bsv0['viewFace'] = viewFace0
    bsv0['viewVertRgb'] = 0.6 * np.ones_like(viewVertWorld0)
    del tmp0, fullVertWorld0, fullFace0, nFace, faceFlag0, fullFaceRgb0, viewVertWorld0, viewFace0

    # world2cam
    for meshName in ['view', 'full']:
        bsv0['%sVertCam' % meshName] = np.matmul(
            bsv0['%sVertWorld' % meshName], camR0.transpose()) + camT0[None, :]

    # doPred
    if ifRequiresPredictingHere:
        # return: predMetricVertCam, predMetricFace, predMetricVertWorld
        bsv0 = kwargs['doPred0Func'](bsv0, **kwargs)

    # predicted rendered depth
    sysLabel = 'cam'
    meshName = 'predMetric'
    camInv0 = bsv0['camInv']
    fxywxy0 = bsv0['fxywxy']
    vert0 = bsv0['%sVert%s' % (meshName, bt(sysLabel))]
    face0 = bsv0['%sFace' % meshName]
    pyrenderManager.clear()
    pyrenderManager.add_camera(fxywxy0, np.eye(4).astype(np.float32))
    pyrenderManager.add_plain_mesh(vert0, face0)
    tmp = pyrenderManager.render()
    d = tmp[1]
    d[(d <= 0) | (d >= 10) | (np.isfinite(d) == 0)] = np.nan
    bsv0['depthRenderedFromPredMetric2'] = d
    bsv0['depthRenderedFromEvalPredMetric2'] = d  # Treat them as the same
    del sysLabel, meshName, camInv0, fxywxy0, vert0, face0, tmp, d

    # polyfit (If the prediction cannot fit, we also won't evaluate their metric benchmarkings, although it exists).
    bsv0['depthForUse2'] = bsv0['depthForUse'][0, :, :]
    a, b, _ = affinePolyfitWithNaN(bsv0['depthRenderedFromEvalPredMetric2'], bsv0['depthForUse2'])
    if np.isnan(a) or np.isnan(b):
        print('[********* Benchmarking Warning *********] fit fails. Always do the manual check'
              'for each such case.')
        if bsv0['dataset'] == 'scannetGivenRender' and bsv0['scanID'] == 'scene0754_00' and bsv0['viewID'] == 1605:
            pass
            # This test case is a flat wall and leads to problematic test space delineation
            # It seems to raise this error on almost all the approaches, and results in NaN benchmarking numbers.
        elif raiseOrTrace == 'trace':
            import ipdb
            ipdb.set_trace()
            print(1 + 1)
        elif raiseOrTrace == 'raise':
            raise ValueError('Fitting a and b becomes NaN.')
        elif raiseOrTrace == 'ignoreAndNone':
            return None
        elif raiseOrTrace == 'ignoreAndNan':
            return exitWithNanBenchmarking(bsv0)
        else:
            raise ValueError('Unknown raiseOrTrace: %s' % raiseOrTrace)
    bsv0['affinePolyfitA'] = float(a)
    bsv0['affinePolyfitB'] = float(b)
    bsv0['depthRenderedFromEvalPredFit2'] = a * bsv0['depthRenderedFromEvalPredMetric2'] + b
    del a, b

    # obtain vertCamPersp
    for meshName in ['predMetric', 'view']:
        fScaleWidth = float(bsv0['fScaleWidth'])
        fScaleHeight = float(bsv0['fScaleHeight'])
        zNear = float(bsv0['zNear'])
        vertCam0 = bsv0['%sVertCam' % meshName]
        bsv0['%sVertCamPersp' % meshName] = np.stack([
            vertCam0[:, 0] * fScaleWidth / np.clip(vertCam0[:, 2], a_min=zNear, a_max=np.inf),
            vertCam0[:, 1] * fScaleHeight / np.clip(vertCam0[:, 2], a_min=zNear, a_max=np.inf),
            vertCam0[:, 2],
        ], 1)
        del fScaleWidth, fScaleHeight, zNear, vertCam0

    # obtain predFitMesh
    for (inputMeshName, outputMeshName) in [('predMetric', 'predFit')]:
        inputVertCamPersp0 = bsv0['%sVertCamPersp' % inputMeshName]
        a, b = bsv0['affinePolyfitA'], bsv0['affinePolyfitB']
        cam0 = bsv0['cam']
        camR0, camT0 = cam0[:3, :3], cam0[:3, 3]
        outputVertCamPersp0 = np.stack([
            inputVertCamPersp0[:, 0], inputVertCamPersp0[:, 1], a * inputVertCamPersp0[:, 2] + b,
        ], 1)
        outputVertCam0 = np.stack([
            outputVertCamPersp0[:, 0] * outputVertCamPersp0[:, 2] / float(bsv0['fScaleWidth']),
            outputVertCamPersp0[:, 1] * outputVertCamPersp0[:, 2] / float(bsv0['fScaleHeight']),
            outputVertCamPersp0[:, 2],
        ], 1)   
        outputVertWorld0 = np.matmul(outputVertCam0 - camT0[None, :], camR0)  # camR0: transpose of inverse
        bsv0['%sVertWorld' % outputMeshName] = outputVertWorld0
        bsv0['%sVertCam' % outputMeshName] = outputVertCam0
        bsv0['%sVertCamPersp' % outputMeshName] = outputVertCamPersp0
        bsv0['%sFace' % outputMeshName] = bsv0['%sFace' % inputMeshName].copy()
        del a, b, cam0, camR0, camT0, outputVertCam0, outputVertCamPersp0, outputVertWorld0, outputMeshName

    # obtain faceVert[SysLabel], faceVertCentroid[SysLabel]
    for meshName in ['view', 'predMetric', 'predFit']:
        for sysLabel in ['world', 'cam', 'camPersp']:
            bsv0['%sFaceVert%s' % (meshName, bt(sysLabel))] = vertInfo2faceVertInfoNP(
                bsv0['%sVert%s' % (meshName, bt(sysLabel))][None], bsv0['%sFace' % meshName][None],
            )[0]
            bsv0['%sFaceCentroid%s' % (meshName, bt(sysLabel))] = \
                bsv0['%sFaceVert%s' % (meshName, bt(sysLabel))].mean(1)

    # We run both the predMetricMesh and the predFitMesh through benchmarking. Only predFitMesh undergo pyrender.
    # In dumpHtmlxxx, dump both, but only html the predFitMesh

    # mesh into evalMesh
    for meshName in ['view', 'predMetric', 'predFit']:
        flagIn0 = (
            (np.abs(bsv0['%sFaceCentroidCamPersp' % meshName][:, 0]) < tightened) &
            (np.abs(bsv0['%sFaceCentroidCamPersp' % meshName][:, 1]) < tightened) &
            (bsv0['%sFaceCentroidCamPersp' % meshName][:, 2] > bsv0['zNear']) &
            (np.all(bsv0['%sFaceCentroidWorld' % meshName] > bsv0['boundMinWorld'][None, :], axis=1)) &
            (np.all(bsv0['%sFaceCentroidWorld' % meshName] < bsv0['boundMaxWorld'][None, :], axis=1)) &
            (np.all(bsv0['%sFaceCentroidCam' % meshName] > bsv0['boundMinCam'][None, :], axis=1)) &
            (np.all(bsv0['%sFaceCentroidCam' % meshName] < bsv0['boundMaxCam'][None, :], axis=1)) 
        )
        if flagIn0.sum() == 0:
            flagIn0[0] = True  # Screw this test case (all-unoccupancy-predicton inside the space of evaluation).
        tmp = trimVertGeneral(
            bsv0['%sVertWorld' % meshName], bsv0['%sFace' % meshName][flagIn0, :],
            {sysLabel: bsv0['%sVert%s' % (meshName, bt(sysLabel))] for sysLabel in ['cam', 'camPersp']},
        )
        bsv0['eval%sVertWorld' % bt(meshName)] = tmp[0]
        bsv0['eval%sFace' % bt(meshName)] = tmp[1]
        for sysLabel in ['cam', 'camPersp']:
            bsv0['eval%sVert%s' % (bt(meshName), bt(sysLabel))] = tmp[2][sysLabel]
        del flagIn0, tmp

    # obtain faceVert[SysLabel], faceVertCentroid[SysLabel] (3 new meshes)
    for meshName in ['evalView', 'evalPredFit', 'evalPredMetric']:
        for sysLabel in ['world', 'cam', 'camPersp']:
            bsv0['%sFaceVert%s' % (meshName, bt(sysLabel))] = vertInfo2faceVertInfoNP(
                bsv0['%sVert%s' % (meshName, bt(sysLabel))][None], bsv0['%sFace' % meshName][None],
            )[0]
            bsv0['%sFaceCentroid%s' % (meshName, bt(sysLabel))] = \
                bsv0['%sFaceVert%s' % (meshName, bt(sysLabel))].mean(1)

    # pc sampling, prepared for benchmarking and floorPlan plotting
    for meshName in ['full', 'evalView', 'evalPredMetric', 'evalPredFit']:
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
        faceCumsumWeight0_thgpu = torch.cumsum(faceArea0_thgpu.double(), dim=0).float()
        # silly torch.cumsum() that numerically random if not put into higher precision (double)
        point0_thgpu, _, _, _, randInfo0_thgpu = mesh_weighted_sampling_given_normal_fixed_rand(
            numMeshSamplingPoint, faceCumsumWeight0_thgpu.detach().clone(), faceVert0_thgpu,
            faceNormal0_thgpu, {}, {})
        bsv0['%sPCxyz%s' % (pcName, bt(sysLabel))] = point0_thgpu.detach().cpu().numpy()
        del pcName, sysLabel, vert0_thgpu, face0_thgpu, faceVert0_thgpu, faceRawNormal0_thgpu
        del faceArea0_thgpu, faceNormal0_thgpu, faceCumsumWeight0_thgpu, point0_thgpu
        del randInfo0_thgpu

    # benchmarking - 3D
    for metricOrFit in ['fit', 'metric']:
        pcName0 = 'evalPred%s' % bt(metricOrFit)
        pcName1 = 'evalView'
        sysLabel = 'world'
        tmp0 = packageCDF1(
            bsv0['%sPCxyz%s' % (pcName0, bt(sysLabel))],
            bsv0['%sPCxyz%s' % (pcName1, bt(sysLabel))],
            prDistThre=prDistThre, cudaDevice=cudaDevice,
        )
        for k in ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']:
            bsv0['finalBen%s%s' % (bt(metricOrFit), bt(k))] = tmp0[k]
        del pcName0, pcName1, sysLabel, tmp0

    # benchmarking - 2.5D
    for metricOrFit in ['fit', 'metric']:
        depthKey0 = 'depthRenderedFromEvalPred%s2' % bt(metricOrFit)
        depthKey1 = 'depthForUse2'
        tmp0 = packageDepth(bsv0[depthKey0], bsv0[depthKey1])
        for k in ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE',
                  'r1', 'r2', 'r3', 'complete']:
            bsv0['finalBen%s%s' % (bt(metricOrFit), bt(k))] = tmp0[k]
        del depthKey0, depthKey1, tmp0, k

    # clone evalView into evalViewMetric and evalViewFit - they are identical wrt vert0/face0, but 
    #   differ in faceRgb0
    for metricOrFit in ['fit', 'metric']:
        for k in ['vertWorld', 'vertCam', 'vertCamPersp', 'face', 'faceCentroidWorld', 'PCxyzWorld']:
            bsv0['evalView%s%s' % (bt(metricOrFit), bt(k))] = bsv0['evalView%s' % bt(k)]

    # coloring evalPredMetric and evalPredFit, as well as evalPredMetric and evalViewFit
    for metricOrFit in ['fit', 'metric']:
        sysLabel = 'world'
        tmp0 = packageCDF1(
            bsv0['evalPred%sFaceCentroid%s' % (bt(metricOrFit), bt(sysLabel))],
            bsv0['evalView%sFaceCentroid%s' % (bt(metricOrFit), bt(sysLabel))],
            prDistThre=prDistThre, cudaDevice=cudaDevice,
        )
        bsv0['evalPred%sFaceRgb' % bt(metricOrFit)] = 0.2 * np.ones(
            (bsv0['evalPred%sFace' % bt(metricOrFit)].shape[0], 3), dtype=np.float32)
        bsv0['evalPred%sFaceRgb' % bt(metricOrFit)][tmp0['distP'] < prDistThre, 2] = 0.6  # light blue precision correct
        bsv0['evalPred%sFaceRgb' % bt(metricOrFit)][tmp0['distP'] >= prDistThre, 0] = 0.6  # red precision wrong
        bsv0['evalView%sFaceRgb' % bt(metricOrFit)] = 0.07 * np.ones(
            (bsv0['evalView%sFace' % bt(metricOrFit)].shape[0], 3), dtype=np.float32)
        bsv0['evalView%sFaceRgb' % bt(metricOrFit)][tmp0['distR'] < prDistThre, 2] = 0.4  # navy recall correct
        bsv0['evalView%sFaceRgb' % bt(metricOrFit)][tmp0['distR'] >= prDistThre, :2] = 0.6  # yellow recall wrong
        del tmp0, sysLabel

    # create evalViewFit2
    # with the geometry the same as evalViewFit, but color to be gray
    # for the purpose of visualization as the "label"
    sysLabel = 'world'
    metricOrFit = 'fit'
    inputMeshName = 'evalView%s' % bt(metricOrFit)
    outputMeshName = 'evalView%s2' % bt(metricOrFit)
    bsv0['%sVert%s' % (outputMeshName, bt(sysLabel))] = bsv0['%sVert%s' % (inputMeshName, bt(sysLabel))].copy()
    bsv0['%sFace' % outputMeshName] = bsv0['%sFace' % inputMeshName].copy()
    bsv0['%sFaceRgb' % outputMeshName] = 0.6 * np.ones_like(bsv0['%sFace' % outputMeshName]).astype(np.float32)
    del sysLabel, metricOrFit, inputMeshName, outputMeshName
    inputPointCloudName = 'evalViewFit'
    outputPointCloudName = 'evalViewFit2'
    sysLabel = 'world'
    bsv0['%sPCxyz%s' % (outputPointCloudName, bt(sysLabel))] = bsv0['%sPCxyz%s' % (inputPointCloudName, bt(sysLabel))].copy()
    del inputPointCloudName, outputPointCloudName, sysLabel

    # rendering
    if ifRequiresDrawing:   
        print('    [benchmarkingGeometry3DHm] Drawing')
        for meshName in ['evalViewFit', 'evalPredFit', 'evalViewFit2']:
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
                (numView, int(bsv0['winHeight']), int(bsv0['winWidth']), 3), dtype=np.float32)
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
                pyrenderManager.add_faceRgb_mesh(
                    bsv0['%sVertWorld' % meshName], bsv0['%sFace' % meshName],
                    bsv0['%sFaceRgb' % meshName])
                tmp = pyrenderManager.render()
                bsv0[prefix + 'ViewColor'][v, :, :, :] = tmp[0].astype(np.float32) / 255.
                '''
                if v == 0:
                    d = tmp[1]
                    d[(d <= 0) | (d >= 10) | (np.isfinite(d) == 0)] = np.nan
                    bsv0['depthRenderedFromEvalPred2'] = d
                '''

            # draw floor plan here
            floorAxis0, floorAxis1 = 0, 1  # world sys floor plan

            def f(ax):
                for mn in ['evalViewFit', meshName]:
                    pp = bsv0['%sPCxyz%s' % (mn, bt(sysLabel))]
                    ax.scatter(
                        pp[:, floorAxis0], pp[:, floorAxis1],
                        c='r' if mn == 'evalPredFit' else 'b',
                        s=0.01 if mn == 'evalPredFit' else 0.003, marker='.')
                e = bsv0[prefix + 'EPick0List']
                l = bsv0[prefix + 'LCenter0']
                for v in range(numView):
                    ax.scatter(e[v, floorAxis0], e[v, floorAxis1], c='k', s=20, marker='o')
                ax.scatter(l[floorAxis0], l[floorAxis1], c='k', s=50, marker='x')
                ax.scatter(e[0, floorAxis0], e[0, floorAxis1], c='k', s=50, marker='o')

            bsv0[prefix + 'FloorPlan'] = getPltDraw(f)

    return bsv0
