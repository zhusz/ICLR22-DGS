# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import math
import copy
from codes_py.toolbox_3D.self_sampling_v1 import mesh_sampling
from codes_py.toolbox_3D.mesh_surgery_v1 import composeSingleShape, combineMultiShapes_withVertRgb, \
    create_lineArrow_mesh
from codes_py.toolbox_3D.representation_v1 import minMaxBound2vox_yxz, voxSdfSign2mesh_mc, \
    voxSdfSign2mesh_skmc, voxSdf2mesh_skmc, voxYXZLabeling2mesh_twelveTriangles
from datasets_registration import datasetRetrieveList
from codes_py.toolbox_3D.draw_mesh_v1 import pickInitialObservationPoints, readjustObservationPoints, \
    elevateObservationPoints, drawVertColoredMesh0_opendrBackend, drawFaceColoredMesh0_nrBackend
from multiprocessing import Pool
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2
from codes_py.toolbox_3D.sdf_from_multiperson_v1 import sdfTHGPU
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSys0, camPerspSys2CamSys0


bt = lambda s: s[0].upper() + s[1:]


class Class_drawVertColoredMesh0_opendrBackend(object):
    def __init__(self,
                 backgroundImg0,
                 focalLengthWidth, focalLengthHeight,
                 L0, U0,
                 vert0, face0, vertRgb0):
        self.backgroundImg0 = backgroundImg0
        self.focalLengthWidth = focalLengthWidth
        self.focalLengthHeight = focalLengthHeight
        self.L0 = L0
        self.U0 = U0
        self.vert0 = vert0
        self.face0 = face0
        self.vertRgb0 = vertRgb0

    def __call__(self, E0):
        return drawVertColoredMesh0_opendrBackend(
            backgroundImg0=self.backgroundImg0,
            focalLengthWidth=self.focalLengthWidth,
            focalLengthHeight=self.focalLengthHeight,
            E0=E0, L0=self.L0, U0=self.U0,
            vert0=self.vert0, face0=self.face0, vertRgb0=self.vertRgb0,
        )


class TemplateVisualStatic(object):
    def __init__(self):
        pass

    # -------------------------- Step Based -------------------------- #
    @staticmethod
    def constructInitialBatchStepVis0(batch_vis, **kwargs):
        iterCount = kwargs['iterCount']
        visIndex = kwargs['visIndex']
        P = kwargs['P']
        D = kwargs['D']
        S = kwargs['S']
        R = kwargs['R']
        methodologyName = '%s%s%s%sI%d' % (P, D, S, R, iterCount)
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] constructInitialBatchStepVis0 (iterCount=%s, P=%s, D=%s, S=%s, R=%s)' %
                  (iterCount, P, D, S, R))
        bsv0 = {
            'iterCount': iterCount,
            'visIndex': visIndex,  # must contain (always 0)
            'P': P, 'D': D, 'S': S, 'R': R,
            'methodologyName': methodologyName,  # must contain
            'index': int(batch_vis['index'][visIndex]),  # must contain
            'did': int(batch_vis['did'][visIndex]),  # must contain (always 0)
            'datasetID': int(batch_vis['datasetID'][visIndex]),  # must contain
            'dataset': datasetRetrieveList[int(batch_vis['datasetID'][visIndex])],  # must contain
            'flagSplit': int(batch_vis['flagSplit'][visIndex]),  # must contain
        }
        return bsv0

    @staticmethod
    def mergeFromBatchVis(bsv0, batch_vis, **kwargs):
        visIndex = bsv0['visIndex']
        existingKeys = ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                        'index', 'did', 'datasetID', 'dataset', 'flagSplit']
        for k in batch_vis.keys():
            if k not in existingKeys:
                assert k not in bsv0.keys()
                if batch_vis[k].ndim > 0:
                    bsv0[k] = batch_vis[k][visIndex]
        return bsv0

    @staticmethod
    def getTestSuiteP():
        raise NotImplementedError
        return os.path.basename(os.path.dirname(__file__))

    @staticmethod
    def getTestSuiteD():
        raise NotImplementedError('This must be defined in D')

    @classmethod
    def getTestSuiteName(cls):
        return cls.getTestSuiteP() + cls.getTestSuiteD() + cls.__name__

    @staticmethod
    def stepRoomcubeGridStandardVersionOccNet(bsv0, **kwargs):
        outputRoomcubeName = kwargs['outputRoomcubeName']
        roomcubeVoxSize = kwargs['roomcubeVoxSize']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubeGridStandardVersionOccNet'
                  '(outputRoomcubeName = %s, roomcubeVoxSize = %s)' %
                  (outputRoomcubeName, roomcubeVoxSize))

        minBound0 = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        maxBound0 = np.array([+0.5, +0.5, +0.5], dtype=np.float32)
        relaxedMinBound0 = 1.01 * minBound0 + (-0.01) * maxBound0
        relaxedMaxBound0 = (-0.01) * minBound0 + 1.01 * maxBound0
        fullQueryVoxXyzWorld, fullQueryGoxyzWorld, fullQuerySCell = minMaxBound2vox_yxz(
            relaxedMinBound0, relaxedMaxBound0, roomcubeVoxSize ** 3
        )
        Ly, Lx, Lz, _ = fullQueryVoxXyzWorld.shape

        fullQueryVoxXyzWorld = fullQueryVoxXyzWorld.reshape((-1, 3))
        cam0 = bsv0['cam']
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]
        fullQueryVoxXyzCam = np.matmul(fullQueryVoxXyzWorld, camR0.transpose()) \
            + camT0[None, :]
        fullQueryVoxXyzCamPersp = camSys2CamPerspSys0(
            fullQueryVoxXyzCam,
            float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
            float(bsv0['winWidth']), float(bsv0['winHeight']),
        )

        fullQueryMaskfloat = np.ones_like(fullQueryVoxXyzWorld[:, 0])  # (Ly*Lx*Lz)
        fullQueryMaskfloat[
            (fullQueryVoxXyzCamPersp[:, 2] <= 0) |
            (fullQueryVoxXyzCamPersp[:, 0] >= 1) | (fullQueryVoxXyzCamPersp[:, 0] <= -1) |
            (fullQueryVoxXyzCamPersp[:, 1] >= 1) | (fullQueryVoxXyzCamPersp[:, 1] <= -1)
        ] = -1.

        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzWorld'] = \
            fullQueryVoxXyzWorld.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzCam'] = \
            fullQueryVoxXyzCam.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzCamPersp'] = \
            fullQueryVoxXyzCamPersp.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeGoxyzWorld'] = fullQueryGoxyzWorld
        bsv0[outputRoomcubeName + 'RoomcubeSCell'] = fullQuerySCell
        bsv0[outputRoomcubeName + 'RoomcubeMaskfloat'] = \
            fullQueryMaskfloat.reshape((Ly, Lx, Lz))
        return bsv0

    @staticmethod
    def stepRoomcubeGridStandardVersionManhattanRoom(bsv0, **kwargs):
        outputRoomcubeName = kwargs['outputRoomcubeName']
        roomcubeVoxSize = kwargs['roomcubeVoxSize']
        minBound0 = kwargs['minBound0']
        maxBound0 = kwargs['maxBound0']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubeGridStandardVersionManhattanRoom'
                  '(outputRoomcubeName = %s, roomcubeVoxSize = %s)' %
                  (outputRoomcubeName, roomcubeVoxSize))

        # minBound0 = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        # maxBound0 = np.array([+0.5, +0.5, +0.5], dtype=np.float32)
        relaxedMinBound0 = 1.01 * minBound0 + (-0.01) * maxBound0
        relaxedMaxBound0 = (-0.01) * minBound0 + 1.01 * maxBound0
        fullQueryVoxXyzWorld, fullQueryGoxyzWorld, fullQuerySCell = minMaxBound2vox_yxz(
            relaxedMinBound0, relaxedMaxBound0, roomcubeVoxSize ** 3
        )
        Ly, Lx, Lz, _ = fullQueryVoxXyzWorld.shape

        fullQueryVoxXyzWorld = fullQueryVoxXyzWorld.reshape((-1, 3))
        cam0 = bsv0['cam']
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]
        fullQueryVoxXyzCam = np.matmul(fullQueryVoxXyzWorld, camR0.transpose()) \
                             + camT0[None, :]
        fullQueryVoxXyzCamPersp = camSys2CamPerspSys0(
            fullQueryVoxXyzCam,
            float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
            float(bsv0['winWidth']), float(bsv0['winHeight']),
        )

        fullQueryMaskfloat = np.ones_like(fullQueryVoxXyzWorld[:, 0])  # (Ly*Lx*Lz)
        fullQueryMaskfloat[
            (fullQueryVoxXyzCamPersp[:, 2] <= 0) |
            (fullQueryVoxXyzCamPersp[:, 0] >= 1) | (fullQueryVoxXyzCamPersp[:, 0] <= -1) |
            (fullQueryVoxXyzCamPersp[:, 1] >= 1) | (fullQueryVoxXyzCamPersp[:, 1] <= -1)
            ] = -1.

        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzWorld'] = \
            fullQueryVoxXyzWorld.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzCam'] = \
            fullQueryVoxXyzCam.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeVoxXyzCamPersp'] = \
            fullQueryVoxXyzCamPersp.reshape((Ly, Lx, Lz, 3))
        bsv0[outputRoomcubeName + 'RoomcubeGoxyzWorld'] = fullQueryGoxyzWorld
        bsv0[outputRoomcubeName + 'RoomcubeSCell'] = fullQuerySCell
        bsv0[outputRoomcubeName + 'RoomcubeMaskfloat'] = \
            fullQueryMaskfloat.reshape((Ly, Lx, Lz))
        return bsv0

    @staticmethod
    def stepCubeGridStandardVersionManhattan(bsv0, **kwargs):
        outputCubeName = kwargs['outputCubeName']
        cubeVoxSize = kwargs['cubeVoxSize']
        cubeType = kwargs['cubeType']  # roomcube or camcube
        minBound0 = kwargs['minBound0']
        maxBound0 = kwargs['maxBound0']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepCubeGridStandardVersionManhattan'
                  '(outputCubeName = %s, cubeVoxSize = %s)' %
                  (outputCubeName, cubeVoxSize))

        # minBound0 = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        # maxBound0 = np.array([+0.5, +0.5, +0.5], dtype=np.float32)
        relaxedMinBound0 = 1.01 * minBound0 + (-0.01) * maxBound0
        relaxedMaxBound0 = (-0.01) * minBound0 + 1.01 * maxBound0

        cam0 = bsv0['cam']
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]
        if cubeType == 'roomcube':  # world sys
            fullQueryVoxXyzWorld, fullQueryGoxyzWorld, fullQuerySCell = minMaxBound2vox_yxz(
                relaxedMinBound0, relaxedMaxBound0, cubeVoxSize ** 3
            )
            Ly, Lx, Lz, _ = fullQueryVoxXyzWorld.shape
            fullQueryVoxXyzWorld = fullQueryVoxXyzWorld.reshape((-1, 3))
            fullQueryVoxXyzCam = np.matmul(fullQueryVoxXyzWorld, camR0.transpose()) \
                                 + camT0[None, :]
        elif cubeType == 'camcube':  # cam sys
            fullQueryVoxXyzCam, fullQueryGoxyzCam, fullQuerySCell = minMaxBound2vox_yxz(
                relaxedMinBound0, relaxedMaxBound0, cubeVoxSize ** 3
            )
            Ly, Lx, Lz, _ = fullQueryVoxXyzCam.shape
            fullQueryVoxXyzCam = fullQueryVoxXyzCam.reshape((-1, 3))
            fullQueryVoxXyzWorld = np.matmul(fullQueryVoxXyzCam - camT0[None, :],
                                             camR0)  # camR0's inv's transpose is camR0 itself
        else:
            raise ValueError('Unknown cubeType: %s' % cubeType)
        fullQueryVoxXyzCamPersp = camSys2CamPerspSys0(
            fullQueryVoxXyzCam,
            float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
            float(bsv0['winWidth']), float(bsv0['winHeight']),
        )

        fullQueryMaskfloat = np.ones_like(fullQueryVoxXyzWorld[:, 0])  # (Ly*Lx*Lz)
        fullQueryMaskfloat[
            (fullQueryVoxXyzCamPersp[:, 2] <= 0) |
            (fullQueryVoxXyzCamPersp[:, 0] >= 1) | (fullQueryVoxXyzCamPersp[:, 0] <= -1) |
            (fullQueryVoxXyzCamPersp[:, 1] >= 1) | (fullQueryVoxXyzCamPersp[:, 1] <= -1)
            ] = -1.

        bsv0[outputCubeName + '%sVoxXyzWorld' % bt(cubeType)] = \
            fullQueryVoxXyzWorld.reshape((Ly, Lx, Lz, 3))
        bsv0[outputCubeName + '%sVoxXyzCam' % bt(cubeType)] = \
            fullQueryVoxXyzCam.reshape((Ly, Lx, Lz, 3))
        bsv0[outputCubeName + '%sVoxXyzCamPersp' % bt(cubeType)] = \
            fullQueryVoxXyzCamPersp.reshape((Ly, Lx, Lz, 3))
        if cubeType == 'roomcube':  # world sys
            bsv0[outputCubeName + '%sGoxyzWorld' % bt(cubeType)] = fullQueryGoxyzWorld
        elif cubeType == 'camcube':  # cam sys
            bsv0[outputCubeName + '%sGoxyzCam' % bt(cubeType)] = fullQueryGoxyzCam
        else:
            raise ValueError('Unknown cubeType: %s' % cubeType)
        bsv0[outputCubeName + '%sSCell' % bt(cubeType)] = fullQuerySCell
        bsv0[outputCubeName + '%sMaskfloat' % bt(cubeType)] = \
            fullQueryMaskfloat.reshape((Ly, Lx, Lz))
        return bsv0

    @staticmethod
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

    @staticmethod
    def stepCamcubeGridStandardVersionCorescene(bsv0, **kwargs):
        # corescene is a generalized version of corenet
        outputCamcubeName = kwargs['outputCamcubeName']
        camcubeVoxSize = kwargs['camcubeVoxSize']  # scalar int  # sometimes it is also named L
        r = kwargs['r']  # for corenet this is 0.5
        cy = kwargs['cy']  # for corenet this is 0
        dz = kwargs['dz']  # for corenet this is f0 / 2
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepCamcubeGridStandardVersionCorenet'
                  '(outputCamcubeName = %s, camcubeVoxSize = %s, r = %.3f, dz = %.5f)' %
                  (outputCamcubeName, camcubeVoxSize, r, dz))

        L = int(camcubeVoxSize)
        xi = np.linspace(-r + r / L, r - r / L, L).astype(np.float32)
        yi = np.linspace(cy -r + r / L, cy + r - r / L, L).astype(np.float32)
        zi = np.linspace(dz + r / L, 2. * r + dz - r / L, L).astype(np.float32)
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
            -r + r / L, cy - r + r / L, dz + r / L
        ], dtype=np.float32)
        bsv0[outputCamcubeName + 'CamcubeSCell'] = np.array(
            [2. * r / L, 2. * r / L, 2. * r / L], dtype=np.float32)
        bsv0[outputCamcubeName + 'CamcubeMaskfloat'] = fullQueryMaskfloat.astype(
            np.float32).reshape((L, L, L))

        return bsv0

    @staticmethod
    def stepCubeGridStandardVersionSimpleCostVolume(bsv0, **kwargs):
        # simpleCostVolume assumes a uniform spanning between zmin and zmax

        outputCubeName = kwargs['outputCubeName']
        cubeVoxSize = kwargs['cubeVoxSize']
        zmin = kwargs['zmin']
        zmax = kwargs['zmax']
        cubeType = kwargs['cubeType']
        assert zmax > zmin >= 0
        assert cubeType in ['camcube', 'simpleraycube']
        f0 = kwargs['f0']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepCubeGridStandardVersionSimpleCostVolume '
                  '(outputCubeName = %s, cubeVoxSize = %s, zmin = %.3f, zmax = %.3f, '
                  'f0 = %.3f, cubeType = %s)' %
                  (outputCubeName, cubeVoxSize, zmin, zmax, f0, cubeType))

        if cubeType == 'simpleraycube':
            minBound0 = np.array([-1, -1, zmin], dtype=np.float32)
            maxBound0 = np.array([1, 1, zmax], dtype=np.float32)
        elif cubeType == 'camcube':
            minBound0 = np.array([-zmax / f0, -zmax / f0, zmin], dtype=np.float32)
            maxBound0 = np.array([zmax / f0, zmax / f0, zmax], dtype=np.float32)
        else:
            raise NotImplementedError('Unknown cubeType: %s' % cubeType)
        relaxedMinBound0 = 1.01 * minBound0 + (-0.01) * maxBound0
        relaxedMaxBound0 = (-0.01) * minBound0 + 1.01 * maxBound0
        fullQueryVoxXyz, fullQueryGoxyz, fullQuerySCell = minMaxBound2vox_yxz(
            relaxedMinBound0, relaxedMaxBound0, cubeVoxSize ** 3
        )
        Ly, Lx, Lz, _ = fullQueryVoxXyz.shape
        fullQueryVoxXyz = fullQueryVoxXyz.reshape((-1, 3))
        cam0 = bsv0['cam']
        camR0 = cam0[:3, :3]
        camT0 = cam0[:3, 3]

        if cubeType == 'camcube':
            fullQueryVoxXyzCam = fullQueryVoxXyz
            fullQueryVoxXyzCamPersp = camSys2CamPerspSys0(
                fullQueryVoxXyzCam,
                float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
                float(bsv0['winWidth']), float(bsv0['winHeight']),
            )
        elif cubeType == 'simpleraycube':
            fullQueryVoxXyzCamPersp = fullQueryVoxXyz
            fullQueryVoxXyzCam = camPerspSys2CamSys0(
                fullQueryVoxXyzCamPersp,
                float(bsv0['focalLengthWidth']), float(bsv0['focalLengthHeight']),
                float(bsv0['winWidth']), float(bsv0['winHeight']),
            )
        else:
            raise NotImplementedError('Unknown cubeType: %s' % cubeType)
        fullQueryVoxXyzWorld = np.matmul(
            fullQueryVoxXyzCam - camT0[None, :], camR0)  # transpose of inverse is itself
        fullQueryMaskfloat = np.ones_like(fullQueryVoxXyzWorld[:, 0])  # (Ly*Lx*Lz)
        fullQueryMaskfloat[
            (fullQueryVoxXyzCamPersp[:, 2] <= zmin) | (fullQueryVoxXyzCamPersp[:, 2] >= zmax) |
            (fullQueryVoxXyzCamPersp[:, 0] >= 1) | (fullQueryVoxXyzCamPersp[:, 0] <= -1) |
            (fullQueryVoxXyzCamPersp[:, 1] >= 1) | (fullQueryVoxXyzCamPersp[:, 1] <= -1)
        ] = -1.

        bsv0['%s%sVoxXyzWorld' % (outputCubeName, bt(cubeType))] = \
            fullQueryVoxXyzWorld.reshape((Ly, Lx, Lz, 3))
        bsv0['%s%sVoxXyzCam' % (outputCubeName, bt(cubeType))] = \
            fullQueryVoxXyzCam.reshape((Ly, Lx, Lz, 3))
        bsv0['%s%sVoxXyzCamPersp' % (outputCubeName, bt(cubeType))] = \
            fullQueryVoxXyzCamPersp.reshape((Ly, Lx, Lz, 3))
        if cubeType == 'camcube':
            bsv0['%s%sGoxyzCam' % (outputCubeName, bt(cubeType))] = fullQueryGoxyz
        elif cubeType == 'simpleraycube':
            bsv0['%s%sGoxyzCamPersp' % (outputCubeName, bt(cubeType))] = fullQueryGoxyz
        else:
            raise NotImplementedError('Unknown cubeType: %s' % cubeType)
        bsv0['%s%sSCell' % (outputCubeName, bt(cubeType))] = fullQuerySCell
        bsv0['%s%sMaskfloat' % (outputCubeName, bt(cubeType))] = fullQueryMaskfloat.reshape(
            (Ly, Lx, Lz))

        return bsv0

    @staticmethod
    def stepCameracubeXyzFromDepthmax(bsv0, **kwargs):
        inplaceCameracubeName = kwargs['inplaceCameracubeName']  # still functional
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepCameracubeXyzFromDepthMax'
                  '(inplaceCameracubeName = %s)' % inplaceCameracubeName)

        cameracubeDepthmax0 = bsv0['%sCameracubeDepthmax' % inplaceCameracubeName]
        lencube0 = bsv0['%sCameracubeLencube' % inplaceCameracubeName]
        Lx, Ly, Lz = lencube0[0], lencube0[1], lencube0[2]
        focalLengthWidth0 = bsv0['focalLengthWidth']
        focalLengthHeight0 = bsv0['focalLengthHeight']
        winWidth0 = bsv0['winWidth']
        winHeight0 = bsv0['winHeight']
        for k in ['VoxXyzWorld', 'VoxXyzCam', 'VoxXyzCamPersp', 'GoxyzCam', 'SCell', 'Maskfloat']:
            assert '%sCameracube%s' % (inplaceCameracubeName, k) not in list(bsv0.keys())
        sCellZ = float(cameracubeDepthmax0 / Lz)
        sCellX = sCellZ * winWidth0 / (2. * focalLengthWidth0)
        sCellY = sCellZ * winHeight0 / (2. * focalLengthHeight0)
        # It means the last depth layer is precisely aligned to the image rendering
        # To make each grid a pure cube (sCellX == sCellY == sCellZ) please try best to let
        # focalLength equals to winSize / 2.
        goxyzCamX = -(Lx / 2. - 0.5) * sCellX
        goxyzCamY = -(Ly / 2. - 0.5) * sCellY
        goxyzCamZ = sCellZ / 2.
        xi = np.linspace(goxyzCamX, goxyzCamX + (Lx - 1) * sCellX, Lx).astype(np.float32)
        yi = np.linspace(goxyzCamY, goxyzCamY + (Ly - 1) * sCellY, Ly).astype(np.float32)
        zi = np.linspace(goxyzCamZ, goxyzCamZ + (Lz - 1) * sCellZ, Lz).astype(np.float32)
        x, y, z = np.meshgrid(xi, yi, zi)  # (stored as YXZ)
        voxXyzCam0 = np.stack([x, y, z], 3).reshape((-1, 3))  # (-1, 3)
        voxXyzCamPersp0 = camSys2CamPerspSys0(
            voxXyzCam0, focalLengthWidth0, focalLengthHeight0,
            winWidth0, winHeight0,
        )  # (-1, 3)
        maskfloat0 = np.ones_like(voxXyzCam0[:, 0])  # (Ly, Lx, Lz)
        maskfloat0[
            (voxXyzCamPersp0[:, 2] <= 0) |
            (voxXyzCamPersp0[:, 0] >= 1) | (voxXyzCamPersp0[:, 0] <= -1) |
            (voxXyzCamPersp0[:, 1] >= 1) | (voxXyzCamPersp0[:, 1] <= -1)
        ] = -1.
        bsv0['%sCameracubeVoxXyzCam' % inplaceCameracubeName] = voxXyzCam0.reshape((Ly, Lx, Lz, 3))
        bsv0['%sCameracubeVoxXyzCamPersp' % inplaceCameracubeName] = \
            voxXyzCamPersp0.reshape((Ly, Lx, Lz, 3))
        bsv0['%sCameracubeGoxyzCam' % inplaceCameracubeName] = \
            np.array([goxyzCamX, goxyzCamY, goxyzCamZ], dtype=np.float32)
        bsv0['%sCameracubeSCell' % inplaceCameracubeName] = \
            np.array([sCellX, sCellY, sCellZ], dtype=np.float32)
        bsv0['%sCameracubeMaskfloat' % inplaceCameracubeName] = maskfloat0.reshape((Ly, Lx, Lz))
        return bsv0

    @staticmethod
    def stepCubeXyzFromGoxyzAndSCellAndLencube(bsv0, **kwargs):
        inplaceCubeName = kwargs['inplaceCubeName']  # still functional
        cubeType = kwargs['cubeType']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepCubeXyzFromGoxyzCamAndSCellAndLencube'
                  '(inplaceCubeName = %s, cubeType = %s)' % (inplaceCubeName, cubeType))

        if cubeType in ['cameracube', 'camcube']:
            sysLabel = 'cam'
        elif cubeType in ['roomcube']:
            sysLabel = 'world'
        else:
            raise NotImplementedError('Unknown cubeType %s' % cubeType)

        focalLengthWidth0 = bsv0['focalLengthWidth']
        focalLengthHeight0 = bsv0['focalLengthHeight']
        winWidth0 = bsv0['winWidth']
        winHeight0 = bsv0['winHeight']
        goxyz = bsv0['%s%sGoxyz%s' % (inplaceCubeName, bt(cubeType), bt(sysLabel))]
        goxyzX, goxyzY, goxyzZ = goxyz[0], goxyz[1], goxyz[2]
        sCell = bsv0['%s%sSCell' % (inplaceCubeName, bt(cubeType))]
        sCellX, sCellY, sCellZ = sCell[0], sCell[1], sCell[2]
        lencube0 = bsv0['%s%sLencube' % (inplaceCubeName, bt(cubeType))]
        Lx, Ly, Lz = lencube0[0], lencube0[1], lencube0[2]
        xi = np.linspace(goxyzX, goxyzX + (Lx - 1) * sCellX, Lx).astype(np.float32)
        yi = np.linspace(goxyzY, goxyzY + (Ly - 1) * sCellY, Ly).astype(np.float32)
        zi = np.linspace(goxyzZ, goxyzZ + (Lz - 1) * sCellZ, Lz).astype(np.float32)
        x, y, z = np.meshgrid(xi, yi, zi)
        voxXyz0 = np.stack([x, y, z], 3).reshape((-1, 3))

        if cubeType in ['cameracube', 'camcube']:
            voxXyzCam0 = voxXyz0
            voxXyzCamPersp0 = camSys2CamPerspSys0(
                voxXyzCam0, focalLengthWidth0, focalLengthHeight0,
                winWidth0, winHeight0,
            )  # (-1, 3)
            maskfloat0 = np.ones_like(voxXyzCam0[:, 0])  # (Ly, Lx, Lz)
            maskfloat0[
                (voxXyzCamPersp0[:, 2] <= 0) |
                (voxXyzCamPersp0[:, 0] >= 1) | (voxXyzCamPersp0[:, 0] <= -1) |
                (voxXyzCamPersp0[:, 1] >= 1) | (voxXyzCamPersp0[:, 1] <= -1)
                ] = -1.
            for k in ['VoxXyzWorld', 'VoxXyzCam', 'VoxXyzCamPersp', 'Maskfloat']:
                assert '%s%s%s' % (inplaceCubeName, bt(cubeType), k) not in list(bsv0.keys())
            bsv0['%s%sVoxXyzCam' % (inplaceCubeName, bt(cubeType))] = \
                voxXyzCam0.reshape((Ly, Lx, Lz, 3))
            bsv0['%s%sVoxXyzCamPersp' % (inplaceCubeName, bt(cubeType))] = \
                voxXyzCamPersp0.reshape((Ly, Lx, Lz, 3))
            bsv0['%s%sMaskfloat' % (inplaceCubeName, bt(cubeType))] = \
                maskfloat0.reshape((Ly, Lx, Lz))
        else:
            raise NotImplementedError('Unknown cubeType: %s' % cubeType)

        return bsv0

    @staticmethod
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

    @staticmethod
    def stepRoomcubeOccfloatToPlainMesh(bsv0, **kwargs):
        inputRoomcubeName = kwargs['inputRoomcubeName']
        outputMeshName = kwargs['outputMeshName']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubeOccfloatToPlainMesh'
                  '(inputRoomcubeName = %s, outputMeshName = %s)' % (
                inputRoomcubeName, outputMeshName))

        occfloat = bsv0[inputRoomcubeName + 'RoomcubeOccfloat'].copy()
        maskFloat = bsv0[inputRoomcubeName + 'RoomcubeMaskfloat'].copy()
        # occfloat[maskFloat <= 0] = 1.  # Unoccupied
        occfloat[maskFloat <= 0] = 1.  # Unoccupied

        # make sure the method can run
        if occfloat.min() >= 0.5:
            occfloat[0, 0, 0] = 0.
        if occfloat.max() <= 0.5:
            occfloat[0, 0, 0] = 1.

        '''
        vertWorld0, face0 = voxSdfSign2mesh_mc(sdfSignFloat,
                                          bsv0[inputRoomcubeName + 'RoomcubeGoxyz'],
                                          bsv0[inputRoomcubeName + 'RoomcubeSCell'])
        '''
        vertWorld0, face0 = voxSdfSign2mesh_skmc(
            occfloat,
            bsv0[inputRoomcubeName + 'RoomcubeGoxyzWorld'],
            bsv0[inputRoomcubeName + 'RoomcubeSCell']
        )
        # Note roomcube must be in the world sys
        bsv0[outputMeshName + 'VertWorld'] = vertWorld0
        bsv0[outputMeshName + 'Face'] = face0

        return bsv0

    @staticmethod
    def stepRoomcubeSdfToPlainMesh(bsv0, **kwargs):
        inputRoomcubeName = kwargs['inputRoomcubeName']
        outputMeshName = kwargs['outputMeshName']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubeSdfToPlainMesh '
                  '(inputRoomcubeName = %s, outputMeshName = %s)' % (
                   inputRoomcubeName, outputMeshName))

        sdf = bsv0[inputRoomcubeName + 'RoomcubeSdf'].copy()
        maskFloat = bsv0[inputRoomcubeName + 'RoomcubeMaskfloat'].copy()
        sdf[maskFloat <= 0] = np.inf  # Unoccupied

        # make sure the method can run
        if sdf.min() >= 0:
            sdf[0, 0, 0] = -np.inf
        if sdf.max() <= 0:
            sdf[0, 0, 0] = np.inf

        vertWorld0, face0 = voxSdf2mesh_skmc(sdf,
                                             bsv0[inputRoomcubeName + 'RoomcubeGoxyzWorld'],
                                             bsv0[inputRoomcubeName + 'RoomcubeSCell'])
        # Note roomcube must be in the world sys
        bsv0[outputMeshName + 'VertWorld'] = vertWorld0
        bsv0[outputMeshName + 'Face'] = face0

        return bsv0

    @staticmethod
    def stepAddGradSdfArrowColoredPointCloudToMesh(bsv0, **kwargs):
        pointCloudName = kwargs['pointCloudName']
        inputMeshName = kwargs['inputMeshName']
        outputMeshName = kwargs['outputMeshName']
        gradSdfKey = kwargs['gradSdfKey']
        lineRadius = kwargs['lineRadius']
        arrowRadius = kwargs['arrowRadius']
        lineLength = kwargs['lineLength']
        sysLabel = kwargs['sysLabel']
        blueGradSdfNormVal = kwargs['blueGradSdfNormVal']
        redGradSdfNormVal = kwargs['redGradSdfNormVal']
        assert redGradSdfNormVal > blueGradSdfNormVal

        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepAddSdfColoredPointCloudToMesh (inputMeshName = %s, '
                  'pointCloudName = %s, outputMeshName = %s, gradSdfKey = %s)' %
                  (inputMeshName, pointCloudName, outputMeshName, gradSdfKey))

        pc = bsv0[pointCloudName + ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))]
        gradSdf = bsv0[pointCloudName + 'PC' + gradSdfKey]
        gradSdfNorm = np.linalg.norm(gradSdf, ord=2, axis=1)
        gradSdfNormalizedAndLengthed = np.divide(
            gradSdf, gradSdfNorm[:, None] + 1.e-6
        ) * lineLength
        redColor = np.array([1., 1., 0.], dtype=np.float32)
        blueColor = np.array([0., 1., 1.], dtype=np.float32)

        def calcColor(gradSdfNormVal):
            alpha = (gradSdfNormVal - blueGradSdfNormVal) / (redGradSdfNormVal - blueGradSdfNormVal)
            alphaTruncated = min(1, max(0, alpha))
            color = (1. - alphaTruncated) * blueColor + alphaTruncated * redColor
            return color

        vfcList = [
            create_lineArrow_mesh(
                rLine=lineRadius,
                rArrow=arrowRadius,
                origin_xyz=pc[p, :],
                terminal_xyz=pc[p, :] + gradSdfNormalizedAndLengthed[p, :],
                color=calcColor(gradSdfNorm[p]),
            )
            for p in range(pc.shape[0])
        ]
        vfcList.append((
            bsv0[inputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
            bsv0[inputMeshName + 'Face'],
            bsv0[inputMeshName + 'VertRgb'],
        ))
        vfc = combineMultiShapes_withVertRgb(vfcList)
        bsv0[outputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))] = vfc[0]
        bsv0[outputMeshName + 'Face'] = vfc[1]
        bsv0[outputMeshName + 'VertRgb'] = vfc[2]
        return bsv0

    @staticmethod
    def stepAddPointCloudToMesh(bsv0, **kwargs):
        pointCloudName = kwargs['pointCloudName']
        inputMeshName = kwargs['inputMeshName']
        outputMeshName = kwargs['outputMeshName']
        sysLabel = kwargs['sysLabel']
        pointColor = kwargs['pointColor']
        pointShapeName = kwargs['pointShapeName']
        pointRadius = kwargs['pointRadius']

        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepAddPointCloudToMesh (inputMeshName = %s, '
                  'pointCloudName = %s, outputMeshName = %s)' %
                  (inputMeshName, pointCloudName, outputMeshName))

        pc = bsv0[pointCloudName + ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))]
        vfcList = [
            composeSingleShape(pc[p, :], radius=pointRadius, color=pointColor, shapeName=pointShapeName)
            for p in range(pc.shape[0])]
        vfcList.append((
            bsv0[inputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
            bsv0[inputMeshName + 'Face'],
            bsv0[inputMeshName + 'VertRgb'],
        ))
        vfc = combineMultiShapes_withVertRgb(vfcList)
        bsv0[outputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))] = vfc[0]
        bsv0[outputMeshName + 'Face'] = vfc[1]
        bsv0[outputMeshName + 'VertRgb'] = vfc[2]
        return bsv0

    @staticmethod
    def stepPointCloudMergeFromList(bsv0, **kwargs):
        inputPointCloudNameList = kwargs['inputPointCloudNameList']
        outputPointCloudName = kwargs['outputPointCloudName']
        keyList = kwargs['keyList']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPointCloudMergeFromList (outputPointCloudName = %s)' %
                  outputPointCloudName)
        for k in keyList:
            bsv0['%sPC%s' % (outputPointCloudName, k)] = np.concatenate([
                bsv0['%sPC%s' % (inputPointCloudName, k)]
                for inputPointCloudName in inputPointCloudNameList
            ], 0)
        return bsv0

    @staticmethod
    def stepPointCloudSplitToList(bsv0, **kwargs):
        inputPointCloudName = kwargs['inputPointCloudName']
        outputPointCloudNameIndDict = kwargs['outputPointCloudNameIndDict']  # key: name, val: ind of whole
        keyList = kwargs['keyList']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPointCloudSplitToList (inputPointCloudName = %s, '
                  'outputPointCloudNameIndDict.keys() == %s' %
                  (inputPointCloudName, list(outputPointCloudNameIndDict.keys())))
        for k in keyList:
            for outputPointCloudName in outputPointCloudNameIndDict.keys():
                ind = outputPointCloudNameIndDict[outputPointCloudName]
                bsv0['%sPC%s' % (outputPointCloudName, k)] = \
                    bsv0['%sPC%s' % (inputPointCloudName, k)][ind]
        return bsv0

    @staticmethod
    def stepAddSdfColoredPointCloudToMesh(bsv0, **kwargs):
        pointCloudName = kwargs['pointCloudName']
        inputMeshName = kwargs['inputMeshName']
        outputMeshName = kwargs['outputMeshName']
        sdfKey = kwargs['sdfKey']
        sysLabel = kwargs['sysLabel']
        pointRadius = kwargs['pointRadius']
        blueSdfVal = kwargs['blueSdfVal']
        redSdfVal = kwargs['redSdfVal']
        # assert sysLabel == 'world'
        assert redSdfVal > blueSdfVal

        # Color mapping rule: uniform within [blueSdfVal, redSdfVal]. Elsewhere truncated
        # Shape mapping rule: within [blueSdfVal, redSdfVal], 'sphere' (+) or 'cube' (-)
        # le blueSdfVal or ge redSdfVal: 'tetrahedron'

        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepAddSdfColoredPointCloudToMesh (inputMeshName = %s, '
                  'pointCloudName = %s, outputMeshName = %s, sdfKey = %s)' %
                  (inputMeshName, pointCloudName, outputMeshName, sdfKey))

        pc = bsv0[pointCloudName + ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))]
        sdf = bsv0[pointCloudName + 'PC' + sdfKey]
        redColor = np.array([1., 0., 0.], dtype=np.float32)
        blueColor = np.array([0., 0., 1.], dtype=np.float32)

        def calcColor(sdfVal):
            alpha = (sdfVal - blueSdfVal) / (redSdfVal - blueSdfVal)
            alphaTruncated = min(1, max(0, alpha))
            color = (1. - alphaTruncated) * blueColor + alphaTruncated * redColor
            return color

        def calcShape(sdfVal):
            alpha = (sdfVal - blueSdfVal) / (redSdfVal - blueSdfVal)
            if alpha < 0 or alpha > 1:
                return 'tetrahedron'
            else:
                if sdfVal <= 0:
                    return 'cube'
                else:
                    return 'sphere'

        vfcList = [
            composeSingleShape(
                pc[p, :],
                radius=pointRadius,
                color=calcColor(sdf[p]),
                shapeName=calcShape(sdf[p]),
            )
            for p in range(pc.shape[0])]
        vfcList.append((
            bsv0[inputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
            bsv0[inputMeshName + 'Face'],
            bsv0[inputMeshName + 'VertRgb'],
        ))
        vfc = combineMultiShapes_withVertRgb(vfcList)
        bsv0[outputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))] = vfc[0]
        bsv0[outputMeshName + 'Face'] = vfc[1]
        bsv0[outputMeshName + 'VertRgb'] = vfc[2]
        return bsv0

    @staticmethod
    def stepAddOccfloatColoredPointCloudToMesh(bsv0, **kwargs):
        pointCloudName = kwargs['pointCloudName']
        inputMeshName = kwargs['inputMeshName']
        outputMeshName = kwargs['outputMeshName']
        occfloatKey = kwargs['occfloatKey']
        sysLabel = kwargs['sysLabel']
        pointRadius = kwargs['pointRadius']

        # Color mapping rule:
        # Here we assume occfloat: (0.5, 1] to be red, 0.5 to be yellow, [0, 0.5) to be blue
        # All are sphere

        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepAddOccfloatColoredPointCloudToMesh (inputMeshName = %s, '
                  'pointCloudName = %s, outputMeshName = %s, sdfKey = %s)' %
                  (inputMeshName, pointCloudName, outputMeshName, occfloatKey))

        pc = bsv0[pointCloudName + ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))]
        occfloat = bsv0[pointCloudName + 'PC' + occfloatKey]

        def calcColor(occfloatVal):
            if occfloatVal > 0.5:
                return [1, 0, 0]
            elif occfloatVal < 0.5:
                return [0, 0, 1]
            else:
                return [1, 1, 0]

        vfcList = [
            composeSingleShape(
                pc[p, :],
                radius=pointRadius,
                color=calcColor(occfloat[p]),
                shapeName='sphere',
            )
            for p in range(pc.shape[0])
        ]
        vfcList.append((
            bsv0[inputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
            bsv0[inputMeshName + 'Face'],
            bsv0[inputMeshName + 'VertRgb'],
        ))
        vfc = combineMultiShapes_withVertRgb(vfcList)
        bsv0[outputMeshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))] = vfc[0]
        bsv0[outputMeshName + 'Face'] = vfc[1]
        bsv0[outputMeshName + 'VertRgb'] = vfc[2]
        return bsv0

    @staticmethod
    def stepMeshToRoomcubeSdf(bsv0, **kwargs):
        inputMeshName = kwargs['inputMeshName']
        inputRoomcubeName = kwargs['inputRoomcubeName']
        outputRoomcubeName = kwargs['outputRoomcubeName']
        cudaDevice = kwargs['cudaDevice']
        referenceSinglePoint = kwargs.get(
            'referenceSinglePoint',
            np.array([-1., -1., -1.], dtype=np.float32),
        )
        # Note Roomcube is always in the world sys, and we do not need the input "sysLabel"
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepMeshToRoomcubeSdf (inputMeshName = %s, outputRoomcubeName = %s)' %
                  (inputMeshName, outputRoomcubeName))

        Ly, Lx, Lz, d = bsv0['%sRoomcubeVoxXyzWorld' % inputRoomcubeName].shape
        assert d == 3
        pointWorld0 = \
            bsv0['%sRoomcubeVoxXyzWorld' % inputRoomcubeName].reshape(
                (-1, 3)).astype(np.float32)
        vertWorld0 = bsv0['%sVertWorld' % inputMeshName].astype(np.float32)
        face0 = bsv0['%sFace' % inputMeshName].astype(np.int32)
        sdf_thgpu, _, _ = sdfTHGPU(
            torch.from_numpy(pointWorld0[None]).to(cudaDevice),
            torch.from_numpy(vertWorld0[None]).to(cudaDevice),
            torch.from_numpy(face0[None]).to(cudaDevice),
            torch.from_numpy(referenceSinglePoint[None]).to(cudaDevice),
            dimDebuggingInfo=0,
        )
        sdf0 = sdf_thgpu[0, :].detach().cpu().numpy().reshape((Ly, Lx, Lz))

        for k in ['VoxXyzWorld', 'GoxyzWorld', 'SCell', 'Maskfloat']:
            bsv0['%sRoomcube%s' % (outputRoomcubeName, k)] = \
                copy.deepcopy(bsv0['%sRoomcube%s' % (inputRoomcubeName, k)])
        bsv0['%sRoomcubeSdf' % outputRoomcubeName] = sdf0
        bsv0['%sRoomcubeOccfloat' % outputRoomcubeName] = (sdf0 > 0).astype(np.float32)

        return bsv0

    @staticmethod
    def stepMeshPointCloudToSdfOccfloat(bsv0, **kwargs):
        inputMeshName = kwargs['inputMeshName']
        inputPointCloudName = kwargs['inputPointCloudName']
        outputPointCloudName = kwargs['outputPointCloudName']
        sysLabel = kwargs['sysLabel']
        cudaDevice = kwargs['cudaDevice']
        referenceSinglePoint = kwargs.get(
            'referenceSinglePoint',
            np.array([[-1., -1., -1.], [1., 1., -1.], [-1., 1., 1.]], dtype=np.float32),
        )
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepMeshPointCloudToSdfOccfloat ' +
                  '(inputMeshName = %s, inputPointCloudName = %s, outputPointCloudName = %s)' %
                  (inputMeshName, inputPointCloudName, outputPointCloudName,)
                  )

        point0_thgpu = torch.from_numpy(
            bsv0['%sPCxyz%s' % (inputPointCloudName, bt(sysLabel))]
        ).float().to(cudaDevice)
        vert0_thgpu = torch.from_numpy(
            bsv0['%sVert%s' % (inputMeshName, bt(sysLabel))]
        ).float().to(cudaDevice)
        face0_thgpu = torch.from_numpy(
            bsv0['%sFace' % inputMeshName]
        ).int().to(cudaDevice)
        sdf_thgpu, _, _ = sdfTHGPU(
            point0_thgpu[None], vert0_thgpu[None], face0_thgpu[None],
            torch.from_numpy(referenceSinglePoint[None]).to(cudaDevice),
            dimCudaIntBuffer=3,
        )
        sdf0 = sdf_thgpu[0, :].detach().cpu().numpy()

        bsv0['%sPCxyz%s' % (outputPointCloudName, bt(sysLabel))] = \
            copy.deepcopy(bsv0['%sPCxyz%s' % (inputPointCloudName, bt(sysLabel))])
        bsv0['%sPCsdf' % outputPointCloudName] = sdf0
        bsv0['%sPCoccfloat' % outputPointCloudName] = (sdf0 > 0).astype(np.float32)
        return bsv0

    @staticmethod
    def stepPlainMeshToPointCloud(bsv0, **kwargs):
        # You cannot use 'vertDirect' to generate the point cloud, since your vert might contain
        # "external" vertices that face does not refer to (like a room in a house)
        inputMeshName = kwargs['inputMeshName']
        outputPointCloudName = kwargs['outputPointCloudName']
        numPoint = kwargs['numPoint']  # only useful if m2pcMethod is 'samplingGet'
        sysLabel = kwargs['sysLabel']
        cudaDevice = kwargs['cudaDevice']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPlainMeshToPointCloud (inputMeshName = %s, '
                  'outputPointCloudName = %s, numPoint = %s, sysLabel = %s)' %
                  (inputMeshName, outputPointCloudName, numPoint, sysLabel))

        point0_thgpu, pointNormal0_thgpu, _, _ = mesh_sampling(
            num_sampling=numPoint,
            vert0_thgpu=torch.from_numpy(bsv0[inputMeshName + 'Vert%s' %
                                                        (sysLabel[0].upper() + sysLabel[1:])])
                .to(device=cudaDevice),
            face0_thgpu=torch.from_numpy(bsv0[inputMeshName + 'Face'])
                .long().to(device=cudaDevice),
            vertInfo0_thgpu={},
            faceInfo0_thgpu={},
        )
        bsv0[outputPointCloudName + 'PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:])] = \
            point0_thgpu.detach().cpu().numpy()
        bsv0[outputPointCloudName + 'PCnormal%s' % (sysLabel[0].upper() + sysLabel[1:])] = \
            pointNormal0_thgpu.detach().cpu().numpy()
        return bsv0

    @staticmethod
    def stepDrawMeshPackage(bsv0, **kwargs):
        inputMeshName = kwargs['inputMeshName']
        outputMeshDrawingName = kwargs['outputMeshDrawingName']
        usingObsFromMeshDrawingName = kwargs['usingObsFromMeshDrawingName']
        numView = kwargs['numView']
        meshVertInfoFaceInfoType = kwargs['meshVertInfoFaceInfoType']
        sysLabel = kwargs['sysLabel']
        ifUseParallel = kwargs.get('ifUseParallel', True)
        ifReadjustObservationPoints = kwargs.get('ifReadjustObservationPoints', True)
        cudaDevice = kwargs['cudaDevice']

        if kwargs['verboseGeneral']:
            print('[Visualizer] stepDrawMeshPackage(inputMeshName = %s, outputMeshDrawingName = %s,'
                  'numView = %s, meshVertInfoFaceInfoType = %s)' %
                  (inputMeshName, outputMeshDrawingName, numView, meshVertInfoFaceInfoType))

        E0 = bsv0['E' + sysLabel[0].upper() + sysLabel[1:]]
        L0 = bsv0['L' + sysLabel[0].upper() + sysLabel[1:]]
        U0 = bsv0['U' + sysLabel[0].upper() + sysLabel[1:]]
        winHeight0 = int(bsv0['winHeight'])
        winWidth0 = int(bsv0['winWidth'])
        focalLengthWidth0 = bsv0['focalLengthWidth']
        focalLengthHeight0 = bsv0['focalLengthHeight']

        if usingObsFromMeshDrawingName is None:
            amodalDepthMax0 = bsv0['amodalDepthMax']
            EPick0List, LCenter0 = pickInitialObservationPoints(E0, L0, U0, amodalDepthMax0, numView)
            if ifReadjustObservationPoints:
                EPick0List = readjustObservationPoints(
                    E0, L0, U0, EPick0List, LCenter0,
                    bsv0[inputMeshName + 'Vert' +
                                   sysLabel[0].upper() + sysLabel[1:]],
                    bsv0[inputMeshName + 'Face'],
                    cudaDevice=cudaDevice)
            for j in range(1, len(EPick0List)):
                EPick0List[j] = 0.85 * EPick0List[j] + 0.15 * LCenter0
            EPick0List = elevateObservationPoints(EPick0List, LCenter0, U0)
        else:
            EPick0List = bsv0[usingObsFromMeshDrawingName + bt(sysLabel)
                                        + bt(meshVertInfoFaceInfoType) + 'EPick0List']
            LCenter0 = bsv0[usingObsFromMeshDrawingName + bt(sysLabel)
                                      + bt(meshVertInfoFaceInfoType) + 'LCenter0']

        # special proessing: put all the things into rgb (always vert rgb format)
        if meshVertInfoFaceInfoType == 'plain':
            rgb = 0.6 * np.ones((bsv0[inputMeshName + 'Vert' + sysLabel[0].upper() + sysLabel[1:]]
                                 .shape[0], 3), dtype=np.float32)
        elif meshVertInfoFaceInfoType in ['vertRgb', 'spaceSdfSignFloatRgb', 'spaceSdfRgb', 'surfaceNyu40IDRgb']:
            rgb = bsv0[
                inputMeshName + meshVertInfoFaceInfoType[0].upper() + meshVertInfoFaceInfoType[1:]].copy()
            rgb = np.maximum(rgb, 0)
            rgb = np.minimum(rgb, 1)
        elif meshVertInfoFaceInfoType == 'plain2':
            color = 0.6 * np.ones((bsv0[inputMeshName + 'Face'].shape[0], 3), dtype=np.float32)
        elif meshVertInfoFaceInfoType in ['surfaceNyu40IDColor', 'spaceNyu40IDColor']:
            color = bsv0[
                inputMeshName + meshVertInfoFaceInfoType[0].upper() + meshVertInfoFaceInfoType[1:]].copy()
        elif meshVertInfoFaceInfoType in ['surfaceNormalCam']:
            raise NotImplementedError
        else:
            raise NotImplementedError('Unknown meshVertInfoFaceInfoType: %s' % meshVertInfoFaceInfoType)

        if meshVertInfoFaceInfoType in ['plain', 'vertRgb', 'spaceSdfSignFloatRgb', 'spaceSdfRgb',
                                        'surfaceNyu40IDRgb']:  # use opendr as backend
            c = Class_drawVertColoredMesh0_opendrBackend(
                backgroundImg0=np.ones((winHeight0, winWidth0, 3), dtype=np.float32),
                focalLengthWidth=int(focalLengthWidth0), focalLengthHeight=int(focalLengthHeight0),
                # E0=EPick0List[v],
                L0=LCenter0, U0=U0,
                vert0=bsv0[inputMeshName + 'Vert' + sysLabel[0].upper() + sysLabel[1:]],
                face0=bsv0[inputMeshName + 'Face'],
                vertRgb0=rgb,
            )

            if ifUseParallel:
                with Pool(min(16, numView * 2)) as p:
                    meshDrawingList = p.map(c, EPick0List)
            else:
                meshDrawingList = [c(e) for e in EPick0List]

            # vert0 = bsv0[inputMeshName + 'Vert' + sysLabel[0].upper() + sysLabel[1:]]
            # face0 = bsv0[inputMeshName + 'Face']
            # faceColor0 = 0.8 * np.ones((face0.shape[0], 3), dtype=np.float32)
            # meshDrawingList = [drawFaceColoredMesh0_nrBackend(
            #     winWidth0, winHeight0, focalLengthWidth0, focalLengthHeight0,
            #     e, LCenter0, U0,
            #     vert0, face0, faceColor0,
            #     cudaDevice,
            # ) for e in EPick0List]

            meshDrawing = np.stack(meshDrawingList, 0)
        elif meshVertInfoFaceInfoType in [
                'plain2', 'spaceNyu40IDColor', 'surfaceNyu40IDColor']:  # use neural renderer as backend
            meshDrawingList = []
            for v2 in range(numView * 2):
                meshDrawing0 = drawFaceColoredMesh0_nrBackend(
                    winWidth=winWidth0, winHeight=winHeight0,
                    focalLengthWidth=float(focalLengthWidth0), focalLengthHeight=float(focalLengthHeight0),
                    E0=EPick0List[v2], L0=LCenter0, U0=U0,
                    vert0=bsv0[inputMeshName + 'Vert' + sysLabel[0].upper() + sysLabel[1:]],
                    face0=bsv0[inputMeshName + 'Face'],
                    faceColor0=color,
                    cudaDevice=cudaDevice,
                )
                meshDrawingList.append(meshDrawing0)
            meshDrawing = np.stack(meshDrawingList, 0)
        else:
            raise NotImplementedError('Unknown meshVertInfoFaceInfoType: %s' % meshVertInfoFaceInfoType)

        EPick0List = np.array(EPick0List, dtype=np.float32)

        # The following two are wrong. EPick0List / LCenter0 are now abstract
        # (can be either world or cam sys)
        # EPickCam0List = np.matmul(EPick0List, camR0.transpose()) + camT0[None, :]
        # LCenterCam0 = np.dot(camR0, LCenter0) + camT0

        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'NumView'] = numView
        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'EPick0List'] = EPick0List
        '''
        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'EPickCam0List'] = EPickCam0List
        '''
        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'LCenter0'] = LCenter0
        '''
        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'LCenterCam0'] = LCenterCam0
        '''
        bsv0[
            outputMeshDrawingName + sysLabel[0].upper() + sysLabel[1:] + meshVertInfoFaceInfoType[0].upper()
            + meshVertInfoFaceInfoType[1:] + 'Drawing0'] = meshDrawing
        return bsv0

    ''' Under Construction
    @staticmethod
    def stepRoomcubeOccfloatPairToChamfer(bsv0, **kwargs):
        inputPredRoomcubeName = kwargs['inputPredRoomcubeName']
        inputLabelRoomcubeName = kwargs['inputLabelRoomcubeName']
        outputBenName = kwargs['outputBenName']
        sysLabel = kwargs['sysLabel']
        cudaDevice = kwargs['cudaDevice']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubePairToChamfer (outputBenName = %s)' % outputBenName)

        pred = bsv0['%s']

        return bsv0
    '''

    @staticmethod
    def stepPointCloudPairToChamfer(bsv0, **kwargs):
        inputPredPointCloudName = kwargs['inputPredPointCloudName']
        inputLabelPointCloudName = kwargs['inputLabelPointCloudName']
        outputBenName = kwargs['outputBenName']
        sysLabel = kwargs['sysLabel']
        cudaDevice = kwargs['cudaDevice']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPointCloudPairToChamfer (outputBenName = %s)' % outputBenName)

        from pytorch3d.loss.chamfer import chamfer_distance
        chamfer, _ = chamfer_distance(
            torch.from_numpy(
                bsv0['%sPCxyz%s' % (inputPredPointCloudName, bt(sysLabel))][None, :, :],
            ).to(device=cudaDevice),
            torch.from_numpy(
                bsv0['%sPCxyz%s' % (inputLabelPointCloudName, bt(sysLabel))][None, :, :],
            ).to(device=cudaDevice)
        )
        chamfer = float(chamfer)
        bsv0[outputBenName] = chamfer
        if kwargs['verboseGeneral'] > 0:
            print('    %s: %.4f' % (outputBenName, chamfer))
        return bsv0

    @staticmethod
    def stepPointCloudPairToChamferFull(bsv0, **kwargs):
        inputPredPointCloudName = kwargs['inputPredPointCloudName']
        inputLabelPointCloudName = kwargs['inputLabelPointCloudName']
        outputBenName = kwargs['outputBenName']
        sysLabel = kwargs['sysLabel']
        cudaDevice = kwargs['cudaDevice']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPointCloudPairToChamferFull (outputBenName = %s)' % outputBenName)

        # from pytorch3d.loss.chamfer import chamfer_distance
        from codes_py.toolbox_3D.pytorch3d_extension_chamfer_v1 import my_chamfer_distance
        chamfer, _, chamfer_x, _, chamfer_y, _ = my_chamfer_distance(
            torch.from_numpy(
                bsv0['%sPCxyz%s' % (inputPredPointCloudName, bt(sysLabel))][None, :, :],
            ).to(device=cudaDevice),
            torch.from_numpy(
                bsv0['%sPCxyz%s' % (inputLabelPointCloudName, bt(sysLabel))][None, :, :],
            ).to(device=cudaDevice)
        )
        chamfer = float(chamfer)
        bsv0[outputBenName] = chamfer
        bsv0[outputBenName + 'X'] = chamfer_x
        bsv0[outputBenName + 'Y'] = chamfer_y
        if kwargs['verboseGeneral'] > 0:
            print('    %s: %.4f, %s_x: %.4f, %s_y: %.4f' %
                  (outputBenName, chamfer, outputBenName, chamfer_x, outputBenName, chamfer_y))
        return bsv0

    @staticmethod
    def stepRoomcubePairToIou(bsv0, **kwargs):
        inputPredRoomcubeName = kwargs['inputPredRoomcubeName']
        inputLabelRoomcubeName = kwargs['inputLabelRoomcubeName']
        outputBenName = kwargs['outputBenName']
        judgeTag = kwargs['judgeTag']  # 'sdf' or 'occfloat'
        judgeFunc = kwargs['judgeFunc']  # occfloat lambda x: x <= 0.5 sdf lambda x: x < 0
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepRoomcubePairToIou (outputBenName = %s)' % outputBenName)

        assert (bsv0['%sRoomcubeGoxyzWorld' % inputLabelRoomcubeName] ==
                bsv0['%sRoomcubeGoxyzWorld' % inputPredRoomcubeName]).all()
        assert (bsv0['%sRoomcubeSCell' % inputLabelRoomcubeName] ==
                bsv0['%sRoomcubeSCell' % inputPredRoomcubeName]).all()
        maskfloat0 = bsv0['%sRoomcubeMaskfloat' % inputLabelRoomcubeName]
        label = judgeFunc(bsv0['%sRoomcube%s' % (inputLabelRoomcubeName, bt(judgeTag))]) & \
                (maskfloat0 > 0).astype(bool)
        pred = judgeFunc(bsv0['%sRoomcube%s' % (inputPredRoomcubeName, bt(judgeTag))]) & \
               (maskfloat0 > 0).astype(bool)
        if label.sum() + pred.sum() == 0:
            iou0 = 1.
        else:
            iou0 = (label & pred).sum() / (label | pred).sum()

        bsv0[outputBenName] = iou0
        if kwargs['verboseGeneral'] > 0:
            print('    %s: %.4f' % (outputBenName, iou0))
        return bsv0

    @staticmethod
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

    @staticmethod
    def stepPointCloudPairToIou(bsv0, **kwargs):
        inputPredPointCloudName = kwargs['inputPredPointCloudName']
        inputLabelPointCloudName = kwargs['inputLabelPointCloudName']
        outputBenName = kwargs['outputBenName']
        judgeTag = kwargs['judgeTag']  # 'sdf' or 'occfloat'
        judgeFunc = kwargs['judgeFunc']  # occfloat lambda x: x <= 0.5 sdf lambda x: x < 0
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPointCloudPairIou (outputBenName = %s)' % outputBenName)

        label = judgeFunc(bsv0['%sPC%s' % (inputLabelPointCloudName, judgeTag)]).astype(bool)
        pred = judgeFunc(bsv0['%sPC%s' % (inputPredPointCloudName, judgeTag)]).astype(bool)
        if label.sum() + pred.sum() == 0:
            iou0 = 1.
        else:
            iou0 = (label & pred).sum() / (label | pred).sum()

        bsv0[outputBenName] = iou0
        if kwargs['verboseGeneral'] > 0:
            print('    %s: %.4f' % (outputBenName, iou0))
        return bsv0

    @staticmethod
    def htmlAddPointCloudFloorPlanScatterVis(bsv0,
                                             summary, txt0, **kwargs):
        scatterColor = kwargs['scatterColor']
        scatterScale = kwargs['scatterScale']
        pointCloudName = kwargs['pointCloudName']
        sysLabel = kwargs['sysLabel']
        floorPlanAxes = kwargs['floorPlanAxes']
        title = kwargs['title']
        txtMessage = kwargs['txtMessage']
        ifAddEL = kwargs['ifAddEL']
        ifAddViewingPoints = kwargs['ifAddViewingPoints']
        methodologyNickName = kwargs['methodologyNickName']
        if ifAddViewingPoints:
            viewingPointsMeshDrawingName = kwargs['viewingPointsMeshDrawingName']
            viewingPointsMeshVertInfoFaceInfoType = kwargs['viewingPointsMeshVertInfoFaceInfoType']
            viewingPointsWhichHalfViewsToDraw = kwargs['viewingPointsWhichHalfViewsToDraw']
        assert not (ifAddEL and ifAddViewingPoints)
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] htmlAddPointCloudFloorPlanScatterVis (pointCloudName = %s, sysLabel = %s,'
                  'title = %s, txtMessage = %s)' % (pointCloudName, sysLabel, title, txtMessage))

        if floorPlanAxes == 'xy':
            floorPlanAxis0, floorPlanAxis1 = 0, 1
        elif floorPlanAxes == 'xz':
            floorPlanAxis0, floorPlanAxis1 = 0, 2
        elif floorPlanAxes == 'yz':
            floorPlanAxis0, floorPlanAxis1 = 1, 2
        else:
            raise NotImplementedError('Unknown floorPlanAxes: %s' % floorPlanAxes)

        def f(ax):
            ax.scatter(bsv0[pointCloudName +
                                      ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))][:, floorPlanAxis0],
                       bsv0[pointCloudName +
                                      ('PCxyz%s' % (sysLabel[0].upper() + sysLabel[1:]))][:, floorPlanAxis1],
                       s=scatterScale, c=scatterColor)
            if ifAddEL:
                E0 = bsv0['E%s' % (sysLabel[0].upper() + sysLabel[1:])]
                L0 = bsv0['L%s' % (sysLabel[0].upper() + sysLabel[1:])]
                ax.scatter(E0[floorPlanAxis0], E0[floorPlanAxis1], s=50, c='k', marker='o')
                ax.scatter(L0[floorPlanAxis0], L0[floorPlanAxis1], s=50, c='k', marker='x')
            if ifAddViewingPoints:
                numView = int(bsv0['%s%s%sNumView' %
                                             (viewingPointsMeshDrawingName, bt(sysLabel),
                                              bt(viewingPointsMeshVertInfoFaceInfoType))])
                EPick0List = bsv0['%s%s%sEPick0List' %
                                            (viewingPointsMeshDrawingName,
                                             bt(sysLabel),
                                             bt(viewingPointsMeshVertInfoFaceInfoType))]
                LCenter0 = bsv0['%s%s%sLCenter0' %
                                          (viewingPointsMeshDrawingName,
                                           bt(sysLabel),
                                           bt(viewingPointsMeshVertInfoFaceInfoType))]
                ax.scatter(LCenter0[floorPlanAxis0], LCenter0[floorPlanAxis1], s=50, c='k', marker='x')
                for view in range(numView):
                    retrieveIndex = view + numView * viewingPointsWhichHalfViewsToDraw
                    ax.scatter(EPick0List[retrieveIndex, floorPlanAxis0], EPick0List[retrieveIndex, floorPlanAxis1],
                               s=50, c='k', marker='o')

        cloth_xy = getPltDraw(f)
        summary['(%s) %s, coordinate sys: %s %s' %
                (methodologyNickName, title, sysLabel, floorPlanAxes)] = \
            cloth_xy[None]
        txt0.append(txtMessage + '%s axis' % floorPlanAxes)

    @staticmethod
    def htmlAddMeshDrawing(bsv0, summary, txt0, **kwargs):
        # Note: if you also wish to scatter point cloud, you should call the function above, then
        # call this function, and then apply brInds
        meshDrawingName = kwargs['meshDrawingName']
        sysLabel = kwargs['sysLabel']
        meshVertInfoFaceInfoType = kwargs['meshVertInfoFaceInfoType']
        whichHalfViewsToDraw = kwargs['whichHalfViewsToDraw']
        title = kwargs['title']
        txtMessage = kwargs['txtMessage']

        assert whichHalfViewsToDraw in [0, 1]  # 0 is the original selected views, while 1 is elevated view
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] htmlAddMeshDrawing (meshDrawingName = %s, whichHalfViewsToDraw = %s,'
                  'sysLabel = %s, meshVertInfoFaceInfoType = %s, title = %s, txtMessage = %s)' %
                  (meshDrawingName, whichHalfViewsToDraw,
                   sysLabel, meshVertInfoFaceInfoType, title, txtMessage))

        numView = int(bsv0['%s%s%sNumView' %
                                     (meshDrawingName, sysLabel[0].upper() + sysLabel[1:],
                                      meshVertInfoFaceInfoType[0].upper() + meshVertInfoFaceInfoType[1:])])
        for view in range(numView):
            retrieveIndex = view + numView * whichHalfViewsToDraw
            summary['%s_%s%s%s_View%d(Half%s)' % (
                title, meshDrawingName, sysLabel[0].upper() + sysLabel[1:],
                meshVertInfoFaceInfoType[0].upper() + meshVertInfoFaceInfoType[1:],
                view, whichHalfViewsToDraw)] = \
                bsv0['%s%s%sDrawing0' % (
                    meshDrawingName, bt(sysLabel), bt(meshVertInfoFaceInfoType)
                )][retrieveIndex, :, :, :][None]
            txt0.append('')

    @staticmethod
    def dumpMeshAsPly(bsv0, **kwargs):
        visualDir = kwargs['visualDir']
        meshName = kwargs['meshName']
        meshVertInfoFaceInfoType = kwargs['meshVertInfoFaceInfoType']
        sysLabel = kwargs['sysLabel']

        methodologyName = bsv0['methodologyName']
        visIndex = bsv0['visIndex']

        plyFileName = visualDir + 'index_%d_%s_%s_%s_%s_%s(%d)_visIndex_%d.ply' % (
            bsv0['index'],
            meshName,
            sysLabel,
            meshVertInfoFaceInfoType,
            methodologyName,
            bsv0['dataset'],
            bsv0['flagSplit'],
            visIndex,
        )
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] dumpMeshAsPly (plyFileName = %s)' % plyFileName)
        if not (meshName + 'Face') in bsv0.keys():
            open(plyFileName, 'w').close()
            return

        if meshVertInfoFaceInfoType == 'plain':
            rgb = 0.6 * np.ones((bsv0[meshName + 'Vert%s' % (sysLabel[0].upper() + sysLabel[1:])]
                                 .shape[0], 3), dtype=np.float32)  # vertRgb
        else:
            rgb = bsv0[meshName + meshVertInfoFaceInfoType[0].upper() + meshVertInfoFaceInfoType[1:]].copy()
            rgb = np.maximum(rgb, 0)
            rgb = np.minimum(rgb, 1)

        # special processing: rgb is just a temporary variable.
        if meshVertInfoFaceInfoType in ['plain', 'vertRgb', 'spaceSdfSignFloatRgb', 'spaceSdfRgb',
                                        'surfaceNyu40IDRgb']:  # (vert rgb)
            dumpPly(fn=plyFileName,
                    vert0=bsv0[meshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
                    face0=bsv0[meshName + 'Face'],
                    vertRgb0=rgb)
        elif meshVertInfoFaceInfoType in ['surfaceNyu40IDColor', 'spaceNyu40IDColor', 'faceRgb']:  # (face color)
            dumpPly2(fn=plyFileName,
                     vert0=bsv0[meshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
                     face0=bsv0[meshName + 'Face'],
                     faceRgb0=rgb)
        elif meshVertInfoFaceInfoType == 'surfaceNormalCam':  # (face color requires abs)
            dumpPly2(fn=plyFileName,
                     vert0=bsv0[meshName + ('Vert%s' % (sysLabel[0].upper() + sysLabel[1:]))],
                     face0=bsv0[meshName + 'Face'],
                     faceRgb0=rgb)
        else:
            raise NotImplementedError('Unknown meshVertInfoFaceInfoType %s' % meshVertInfoFaceInfoType)
