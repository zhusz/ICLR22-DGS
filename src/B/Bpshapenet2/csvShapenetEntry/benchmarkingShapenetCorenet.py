# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, camSys2CamPerspSys0
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.corenet.geometry.voxelization import voxelize_mesh
from codes_py.corenet.cc import fill_voxels
from codes_py.toolbox_3D.representation_v1 import minMaxBound2vox_yxz, voxSdfSign2mesh_mc, \
    voxSdfSign2mesh_skmc
import numpy as np


bt = lambda s: s[0].upper() + s[1:]


def generalVoxelization(faceVertZO0List, Ly, Lx, Lz, cudaDevice):
    # ZO: Zero To One
    # Note this ZO cam sys is the corenet cube in the cam sys
    # It is different from camTheirs in the _voxelization() method
    # for the y dim they are summed to 1 (complementary)
    # so in this method, we should not reverse the y dim in the output.

    faceVertZO0s = np.concatenate(faceVertZO0List, 0)
    nFaces = np.array([faceVertZO0.shape[0] for faceVertZO0 in faceVertZO0List],
                      dtype=np.int32)
    grids_thcpu = voxelize_mesh(
        torch.from_numpy(faceVertZO0s),
        torch.from_numpy(nFaces),
        resolution=(Lx, Ly, Lz),
        view2voxel=torch.from_numpy(
            np.array([
                [Lx, 0, 0, 0],
                [0, Ly, 0, 0],
                [0, 0, Lz, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32),
        ),
        sub_grid_sampling=False,
        image_resolution_multiplier=8,
        conservative_rasterization=False,
        projection_depth_multiplier=1,
        cuda_device=None,
    )  # (BZYX)
    grids_thgpu = grids_thcpu.to(cudaDevice)
    fill_voxels.fill_inside_voxels_gpu(grids_thgpu, inplace=True)
    return grids_thgpu  # (BZYX) bisem: 1 for the occupied


def stepCubePairToIou(bsv0, **kwargs):
    inputPredCubeName = kwargs['inputPredCubeName']
    inputLabelCubeName = kwargs['inputLabelCubeName']
    cubeType = kwargs['cubeType']
    outputBenName = kwargs['outputBenName']
    labelJudgeTag = kwargs['labelJudgeTag']
    predJudgeTag = kwargs['predJudgeTag']
    judgeFunc = kwargs['judgeFunc']
    if kwargs['verboseGeneral'] > 0:
        print('[Visualizer] stepCubePairToIou (outputBenName = %s)' % outputBenName)

    if cubeType in ['roomcube']:
        sysLabel = 'world'
    elif cubeType in ['camcube']:
        sysLabel = 'cam'
    else:
        raise ValueError('Unknown cubeType: %s' % cubeType)

    for k in ['goxyz%s' % bt(sysLabel), 'sCell']:
        assert (bsv0['%s%s%s' % (inputLabelCubeName, bt(cubeType), bt(k))] ==
                bsv0['%s%s%s' % (inputPredCubeName, bt(cubeType), bt(k))]).all()

    maskfloat = bsv0['%s%sMaskfloat' % (inputLabelCubeName, bt(cubeType))]
    label = judgeFunc(
        bsv0['%s%s%s' % (inputLabelCubeName, bt(cubeType), bt(labelJudgeTag))]).astype(bool) & \
            (maskfloat > 0)
    pred = judgeFunc(
        bsv0['%s%s%s' % (inputPredCubeName, bt(cubeType), bt(predJudgeTag))]).astype(bool) & \
           (maskfloat > 0)
    if label.sum() + pred.sum() == 0:
        iou0 = 1.
    else:
        iou0 = (label & pred).sum() / (label | pred).sum()

    bsv0[outputBenName] = iou0
    if kwargs['verboseGeneral'] > 0:
        print('    %s: %.4f' % (outputBenName, iou0))
    return bsv0


def stepCubeOccfloatToPlainMesh(bsv0, **kwargs):
    cubeType = kwargs['cubeType']  # roomcube or camcube or cameracube (a special case of camcube)
    inputCubeName = kwargs['inputCubeName']
    outputMeshName = kwargs['outputMeshName']
    occfloatKey = kwargs['occfloatKey']
    if kwargs['verboseGeneral'] > 0:
        print('[Visualizer] stepCubeOccfloatToPlainMesh'
              '(inputCubeName = %s, outputMeshName = %s)' % (
                  inputCubeName, outputMeshName))

    occfloat = bsv0['%s%s%s' % (inputCubeName, bt(cubeType), bt(occfloatKey))].copy()
    maskFloat = bsv0['%s%sMaskfloat' % (inputCubeName, bt(cubeType))].copy()
    occfloat[maskFloat <= 0] = 1.  # Unoccupied

    # make sure the method can run
    if occfloat.min() >= 0.5:
        occfloat[0, 0, 0] = 0.
    if occfloat.max() <= 0.5:
        occfloat[0, 0, 0] = 1.

    if cubeType in ['camcube']:  # cam sys
        vertCam0, face0 = voxSdfSign2mesh_skmc(
            occfloat,
            bsv0['%s%sGoxyzCam' % (inputCubeName, bt(cubeType))],
            bsv0['%s%sSCell' % (inputCubeName, bt(cubeType))],
        )
        bsv0[outputMeshName + 'VertCam'] = vertCam0
        bsv0[outputMeshName + 'Face'] = face0
    elif cubeType in ['simpleraycube']:  # camPersp sys
        vertCamPersp0, face0 = voxSdfSign2mesh_skmc(
            occfloat,
            bsv0['%s%sGoxyzCamPersp' % (inputCubeName, bt(cubeType))],
            bsv0['%s%sSCell' % (inputCubeName, bt(cubeType))],
        )
        bsv0[outputMeshName + 'VertCamPersp'] = vertCamPersp0
        bsv0[outputMeshName + 'Face'] = face0
    elif cubeType in ['roomcube']:  # world sys
        vertWorld0, face0 = voxSdfSign2mesh_skmc(
            occfloat,
            bsv0['%s%sGoxyzWorld' % (inputCubeName, bt(cubeType))],
            bsv0['%s%sSCell' % (inputCubeName, bt(cubeType))],
        )
        bsv0[outputMeshName + 'VertWorld'] = vertWorld0
        bsv0[outputMeshName + 'Face'] = face0
    else:
        raise NotImplementedError
    return bsv0


def stepCamcubeGridStandardVersionCorenet(bsv0, **kwargs):
    outputCamcubeName = kwargs['outputCamcubeName']
    camcubeVoxSize = kwargs['camcubeVoxSize']  # scalar int
    f0 = kwargs['f0']
    if kwargs['verboseGeneral'] > 0:
        print('[Visualizer] stepCamcubeGridStandardVersionCorenet'
              '(outputCamcubeName = %s, camcubeVoxSize = %s)' %
              (outputCamcubeName, camcubeVoxSize))

    L = int(camcubeVoxSize)
    xi = np.linspace(-0.5 + 0.5 / L, 0.5 - 0.5 / L, L).astype(np.float32)
    yi = np.linspace(-0.5 + 0.5 / L, 0.5 - 0.5 / L, L).astype(np.float32)
    zi = np.linspace(f0 / 2 + 0.5 / L, 1. + f0 / 2. - 0.5 / L, L).astype(np.float32)
    x, y, z = np.meshgrid(xi, yi, zi)  # stored as YXZ
    fullQueryVoxXyzCam = np.stack([x, y, z], 3).reshape((-1, 3))
    fullQueryVoxXyzCamPersp = camSys2CamPerspSys0(
        fullQueryVoxXyzCam,
        float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
        float(bsv0['winWidth']), float(bsv0['winHeight']),
    )
    fullQueryMaskfloat = np.ones_like(fullQueryVoxXyzCam[:, 0])  # (Ly*Lx*Lz)
    fullQueryMaskfloat[
        (fullQueryVoxXyzCamPersp[:, 2] <= 0) |
        (fullQueryVoxXyzCamPersp[:, 0] >= 1) | (fullQueryVoxXyzCamPersp[:, 0] <= -1) |
        (fullQueryVoxXyzCamPersp[:, 1] >= 1) | (fullQueryVoxXyzCamPersp[:, 1] <= -1)
        ] = -1.

    bsv0[outputCamcubeName + 'CamcubeVoxXyzCam'] = fullQueryVoxXyzCam.reshape((L, L, L, 3))
    bsv0[outputCamcubeName + 'CamcubeVoxXyzCamPersp'] = \
        fullQueryVoxXyzCamPersp.reshape((L, L, L, 3))
    bsv0[outputCamcubeName + 'CamcubeGoxyzCam'] = np.array([
        -0.5 + 0.5 / L, -0.5 + 0.5 / L, f0 / 2 + 0.5 / L
    ], dtype=np.float32)
    bsv0[outputCamcubeName + 'CamcubeSCell'] = np.array(
        [1. / L, 1. / L, 1. / L], dtype=np.float32)
    bsv0[outputCamcubeName + 'CamcubeMaskfloat'] = fullQueryMaskfloat.astype(
        np.float32).reshape((L, L, L))

    return bsv0


def benchmarkingShapenetCorenetFunc(bsv0, **kwargs):
    datasetObj = kwargs['datasetObj']
    meta = kwargs['meta']

    # prediting
    ifRequiresPredictingHere = kwargs['ifRequiresPredictingHere']

    # drawing
    ifRequiresDrawing = kwargs['ifRequiresDrawing']

    # from bsv0
    index = int(bsv0['index'])

    # from meta
    f0 = meta['f0']

    # Related to benchmarking scores
    camcubeVoxSize = kwargs['camcubeVoxSize']
    cudaDevice = kwargs['cudaDevice']

    bsv0 = stepCamcubeGridStandardVersionCorenet(
        bsv0, outputCamcubeName='corenet', camcubeVoxSize=camcubeVoxSize,
        f0=f0, verboseGeneral=False,
    )

    if ifRequiresPredictingHere:
        # return: corenetCamcubePredOccfloatYXZ
        bsv0 = kwargs['doPred0Func'](bsv0, **kwargs)
    else:
        pass

    # load label mesh
    tmp1 = datasetObj.getRawMeshCamOurs({}, index)
    bsv0['rawVertCam'] = tmp1['vertCamOurs'].astype(np.float32)
    bsv0['rawFace'] = tmp1['face'].astype(np.int32)
    bsv0['rawVertRgb'] = np.ones_like(bsv0['rawVertCam']) * 0.6

    # voxelization label mesh
    bsv0['corenetCamcubeLabelBisemZYX'] = generalVoxelization(
        [vertInfo2faceVertInfoNP(
            np.stack([
                bsv0['rawVertCam'][:, 0] + 0.5,
                bsv0['rawVertCam'][:, 1] + 0.5,
                bsv0['rawVertCam'][:, 2] - f0 / 2.,
            ], 1)[None, :, :],
            bsv0['rawFace'][None, :, :],
        )[0]],
        camcubeVoxSize,
        camcubeVoxSize,
        camcubeVoxSize,
        cudaDevice,
    )[0].detach().cpu().numpy()
    bsv0['corenetCamcubeLabelOccfloatZYX'] = 1. - bsv0['corenetCamcubeLabelBisemZYX']
    bsv0['corenetCamcubeLabelOccfloatYXZ'] = bsv0['corenetCamcubeLabelOccfloatZYX'].transpose(
        (1, 2, 0))
    bsv0['corenetCamcubeLabelBisemYXZ'] = bsv0['corenetCamcubeLabelBisemZYX'].transpose(
        (1, 2, 0))

    # benchmarking Iou computation
    bsv0 = stepCubePairToIou(
        bsv0, inputPredCubeName='corenet', inputLabelCubeName='corenet',
        cubeType='camcube', outputBenName='corenetCubeIou',
        predJudgeTag='predOccfloatYXZ', labelJudgeTag='labelOccfloatYXZ',
        judgeFunc=lambda x: (x < 0.5).astype(bool),
        verboseGeneral=False,
    )
    bsv0['finalBenCorenetCubeIou'] = bsv0['corenetCubeIou']

    if ifRequiresDrawing:
        bsv0 = stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet', outputMeshName='corenetPred',
            occfloatKey='predOccfloatYXZ', verboseGeneral=False,
        )
        bsv0 = stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet',
            outputMeshName='corenetLabel',
            occfloatKey='labelOccfloatYXZ', verboseGeneral=False,
        )
        for meshName in ['corenetPred', 'corenetLabel']:
            sysLabel = 'cam'
            bsv0['%sVertRgb' % meshName] = \
                0.6 * np.ones_like(bsv0['%sVert%s' % (meshName, bt(sysLabel))])

    return bsv0

