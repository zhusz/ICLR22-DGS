# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
import torch
from .dumpCsvGeometry3D import abTheMeshInTheWorldSys, abTheMeshInTheCamSys, \
    judgeWhichTrianglesFlagIn
import numpy as np
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert
from codes_py.toolbox_3D.benchmarking_v1 import packageCDF1, packageDepth, affinePolyfitWithNaN
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP, vertInfo2faceVertInfoTHGPU
from codes_py.toolbox_3D.self_sampling_v1 import mesh_weighted_sampling_given_normal_fixed_rand
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.draw_mesh_v1 import pickInitialObservationPoints, getUPick0List
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, getRotationMatrixBatchNP
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, to_heatmap
# easy debugging
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from skimage.io import imsave
from codes_py.np_ext.mat_io_v1 import pSaveMat
np.set_printoptions(suppress=True)


bt = lambda s: s[0].upper() + s[1:]


def benchmarkingGeometry3DDemoFunc(bsv0, **kwargs):
    datasets = kwargs['datasets']
    raiseOrTrace = kwargs['raiseOrTrace']
    assert raiseOrTrace in ['raise', 'trace']

    # drawing
    ifRequiresDrawing = kwargs['ifRequiresDrawing']
    pyrenderManager = kwargs['pyrenderManager']

    # predicting
    ifRequiresPredictingHere = kwargs['ifRequiresPredictingHere']

    # Related to benchmarking scores
    # (although there is not label, they will determine the prediction process)
    voxSize = kwargs['voxSize']
    numMeshSamplingPoint = kwargs['numMeshSamplingPoint']
    cudaDevice = kwargs['cudaDevice']

    tightened = 1. - 4. / voxSize
    prDistThre = 0.05

    # from bsv0
    index = int(bsv0['index'])
    did = int(bsv0['did'])
    datasetObj = datasets[did]
    dataset = datasetObj.datasetConf['dataset']
    assert dataset.startswith('demo') or dataset.startswith('freedemo') or dataset == 'pix3d'
    datasetConf = datasetObj.datasetConf

    # cam
    bsv0['cam'] = np.eye(4).astype(np.float32)
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

    bsv0['boundMinCam'] = np.array([-3, -3, 0], dtype=np.float32)
    bsv0['boundMaxCam'] = np.array([+3, +3, 6], dtype=np.float32)
    bsv0['boundMinWorld'] = bsv0['boundMinCam']
    bsv0['boundMaxWorld'] = bsv0['boundMaxCam']

    # predicting
    if ifRequiresPredictingHere:
        bsv0 = kwargs['doPred0Func'](bsv0, **kwargs)
    else:
        import ipdb
        ipdb.set_trace()
        print(1 + 1)

    # predicted rendered depth
    # sysLabel = 'world'
    sysLabel = 'cam'
    # meshName = 'evalPred'
    meshName = 'pred'
    camInv0 = bsv0['camInv']
    fxywxy0 = bsv0['fxywxy']
    vert0 = bsv0['%sVert%s' % (meshName, bt(sysLabel))]
    face0 = bsv0['%sFace' % meshName]
    pyrenderManager.clear()
    # pyrenderManager.add_camera(fxywxy0, camInv0)
    pyrenderManager.add_camera(fxywxy0, np.eye(4).astype(np.float32))
    pyrenderManager.add_plain_mesh(vert0, face0)
    tmp = pyrenderManager.render()
    d = tmp[1]
    d[(d <= 0) | (d >= 10) | (np.isfinite(d) == 0)] = np.nan
    bsv0['depthRenderedFromPred2'] = d
    bsv0['depthRenderedFromEvalPred2'] = d  # Treat them as the same

    # polyfit
    bsv0['affinePolyfitA'], bsv0['affinePolyfitB'] = 1, 0
    del sysLabel, meshName, camInv0, fxywxy0, vert0, face0, tmp, d

    # trim away frustum border mesh triangles for the purpose of evaluation
    for meshName in ['pred']:
        vertCam0 = bsv0['%sVertCam' % meshName].copy()
        face0 = bsv0['%sFace' % meshName].copy()
        flagIn = judgeWhichTrianglesFlagIn(
            vertCam0, face0, bsv0['cam'][:3, :3], bsv0['cam'][:3, 3],
            bsv0['fScaleWidth'], bsv0['fScaleHeight'], bsv0['zNear'],
            tightened, bsv0['boundMaxWorld'], bsv0['boundMinWorld'],
        )
        if flagIn.sum() == 0:  # Now this should never happens (unless no mesh error)
            # When there is mesh (i.e. face0.shape[0] > 10) this should never happen
            flagIn[0] = True
            bsv0['allowNaN'] = True

            if not np.all(np.isnan(bsv0['depthRenderedFromPred2'][5:-5, 5:-5])):  # filter out true empty
                # print('[********* Visualizer Warning *********] Empty prediction occurs!')
                print('Now this should never happen until the mesh prediction is empty')
                print('Let this thread go / continue to run only if it is indeed an empty mesh')
                if raiseOrTrace == 'trace':
                    import ipdb
                    ipdb.set_trace()
                    print(1 + 1)
                elif raiseOrTrace == 'raise':
                    raise ValueError('No triangle in the prediction.')
                else:
                    raise ValueError('Unknown raiseOrTrace: %s' % raiseOrTrace)
        else:
            bsv0['allowNan'] = False

        face0 = face0[flagIn]
        vertCam0, face0 = trimVert(vertCam0, face0)
        bsv0['eval%sVertCam' % bt(meshName)] = vertCam0
        bsv0['eval%sFace' % bt(meshName)] = face0
        nFace = face0.shape[0]
        bsv0['eval%sFaceRgb' % bt(meshName)] = np.tile(
            np.array([0.07, 0.07, 0.4], dtype=np.float32)[None, :],
            (nFace, 1),
        )

    # cam2world
    bsv0['EWorld'] = np.array([0, 0, 0], dtype=np.float32)
    bsv0['LWorld'] = np.array([0, 0, 1], dtype=np.float32)
    bsv0['UWorld'] = np.array([0, -1, 0], dtype=np.float32)
    if dataset == 'pix3d':
        bsv0['depthMax'] = 4.
    else:
        bsv0['depthMax'] = 5.
    for meshName in ['evalPred']:
        bsv0['%sVertWorld' % meshName] = np.matmul(
            bsv0['%sVertCam' % meshName] - camT0[None, :], camR0  # transpose of inverse
        )

    # get face centroid
    #   naming convention: vXxxWorld, fcXxxWorld are vert/faceCentroid PC of mesh xxx
    #   without prefix: it is actually the standard numPoint sampling result
    for meshName in ['evalPred']:
        sysLabel = 'world'
        faceCentroid0 = vertInfo2faceVertInfoNP(
            bsv0['%sVert%s' % (meshName, bt(sysLabel))][None],
            bsv0['%sFace' % meshName][None])[0].mean(1)
        bsv0['fc%sPCxyz%s' % (bt(meshName), bt(sysLabel))] = faceCentroid0
        bsv0['v%sPCxyz%s' % (bt(meshName), bt(sysLabel))] = \
            bsv0['%sVert%s' % (meshName, bt(sysLabel))].copy()

    # pc sampling, for benchmarking and visualization
    for meshName in ['evalPred']:
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
        point0_thgpu, _, _, _, randInfo0_thgpu = mesh_weighted_sampling_given_normal_fixed_rand(
            numMeshSamplingPoint, faceCumsumWeight0_thgpu.detach().clone(), faceVert0_thgpu,
            faceNormal0_thgpu, {}, {})
        bsv0['%sPCxyz%s' % (pcName, bt(sysLabel))] = point0_thgpu.detach().cpu().numpy()

    # draw the mesh
    # (even if not drawing required, you still need to render once for depthRendered)
    if ifRequiresDrawing:
        for meshName in ['evalPred']:
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            prefix = '%s%s%sMeshDrawingPackage' % (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))

            numView = 6
            ungrav0 = np.array([0, -1, 0], dtype=np.float32)
            EPick0List, LCenter0 = pickInitialObservationPoints(
                bsv0['EWorld'], bsv0['LWorld'], ungrav0,
                bsv0['depthMax'], numView=9,
            )
            EPick0List = EPick0List[[0, 1, 8], :]
            EPick0ListElevated = EPick0List + 1.2 * ungrav0[None, :]
            EPick0List = np.concatenate([EPick0List, EPick0ListElevated], 0)
            LCenter0 = 1.5 * LCenter0 - 0.5 * bsv0['EWorld']
            bsv0[prefix + 'NumView'] = numView
            bsv0[prefix + 'EPick0List'] = EPick0List
            bsv0[prefix + 'LCenter0'] = LCenter0
            bsv0[prefix + 'UPick0List'] = np.tile(ungrav0[None, :], (numView, 1))
            bsv0[prefix + 'UPick0List'][:, :] = bsv0['UWorld']
            bsv0[prefix + 'ViewColor'] = np.ones(
                (numView, bsv0['winHeight'], bsv0['winWidth'], 3), dtype=np.float32)
            for v in range(numView):
                pyrenderManager.clear()
                pyrenderManager.add_point_light(
                    pointLoc=1.2 * EPick0List[v] - 0.2 * LCenter0, intensity=0.8,
                    color=[200., 200., 200.]
                )
                viewCam0 = ELU02cam0(np.concatenate(
                    [EPick0List[v], LCenter0, bsv0[prefix + 'UPick0List'][v]], 0))
                viewCamInv0 = np.linalg.inv(viewCam0)
                pyrenderManager.add_camera(bsv0['fxywxy'], viewCamInv0)
                pyrenderManager.add_vertRgb_mesh_via_faceRgb(
                    bsv0['%sVertWorld' % meshName], bsv0['%sFace' % meshName],
                    bsv0['%sFaceRgb' % meshName],
                )
                tmp = pyrenderManager.render()
                bsv0[prefix + 'ViewColor'][v, :, :, :] = tmp[0].astype(np.float32) / 255.
                if v == 0:
                    bsv0['depthRenderedFromEvalPred2'] = tmp[1]
            # draw floor plan here
            floorAxis0, floorAxis1 = 0, 2  # world sys floor plan
            def f(ax):
                for mn in [meshName]:
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

    # pseudo
    bsv0['houseID'] = -1
    bsv0['viewID'] = -1

    return bsv0