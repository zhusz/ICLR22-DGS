# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotations_v1 import getRotationMatrixBatchNP, ELU02cam0, camSys2CamPerspSys0
from .mesh_v1 import zBufferBatchNP, zBufferBatchTHGPU, \
    getEdgeMaskTHGPU, placeFaceInfoOntoClothTHGPU
from matplotlib.cm import get_cmap
# from opendr.camera import ProjectPoints
# from opendr.renderer import ColoredRenderer, TexturedRenderer
# from opendr.lighting import LambertianPointLight
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import numpy as np
import easydict


def drawFaceColoredMesh0_nrBackend(winWidth, winHeight, focalLengthWidth, focalLengthHeight,
                                   E0, L0, U0,
                                   vert0, face0, faceColor0,
                                   cudaDevice):
    # Note E / L / U / vert0 should be in the same coord sys, world or cam
    assert len(E0) == 3
    assert len(L0) == 3
    assert len(U0) == 3
    assert len(vert0.shape) == 2 and vert0.shape[1] == 3
    assert len(face0.shape) == 2 and face0.shape[1] == 3
    nVert = vert0.shape[0]

    # rotate everything
    cam0 = ELU02cam0(np.concatenate((E0, L0, U0), 0))
    R0 = cam0[:3, :3]
    Rtransposed0 = R0.transpose()
    T0 = cam0[:3, 3]
    vertNow0 = np.matmul(vert0, Rtransposed0) + T0[None]

    face0_th = torch.from_numpy(face0).to(cudaDevice)
    faceColor0_th = torch.from_numpy(faceColor0).to(cudaDevice)

    winSize = max(winWidth, winHeight)
    vertNowPersp0 = camSys2CamPerspSys0(vertNow0,
                                        focalLengthWidth, focalLengthHeight, winSize, winSize)
    vertNowPersp0_th = torch.from_numpy(vertNowPersp0).to(cudaDevice)
    # _, trID_th, _ = mesh_v1.obtainTrisCoverIDBatchTHGPU(vertNowPersp0_th[None], face0_th[None], winSize)
    _, trID_th, _ = zBufferBatchTHGPU(vertNowPersp0_th[None], face0_th[None], winSize)
    # edgeMask_th = mesh_v1.getEdgeMaskTHGPU(vertNowPersp0_th[None, :, :2], face0_th[None],
    #                                        trID_th, False, True, 0.0015)
    edgeMask_th = getEdgeMaskTHGPU(
        vertNowPersp0_th[None, :, :2], face0_th[None], trID_th, False, True, 0.0015
    )
    starting = int((winSize - min(winHeight, winWidth)) / 2.)
    ending = winSize - starting
    if winHeight < winWidth:
        trID_th = trID_th[:, starting:ending, :]
        edgeMask_th = edgeMask_th[:, starting:ending, :]
    else:
        trID_th = trID_th[:, :, starting:ending]
        edgeMask_th = edgeMask_th[:, :, starting:ending]
    assert trID_th.shape[1] == winHeight and trID_th.shape[2] == winWidth
    cloth_th = placeFaceInfoOntoClothTHGPU(
        trID_th, faceColor0_th[None],
        background=1.)
    cloth_th = torch.clamp(cloth_th - 0.3 * edgeMask_th[:, :, :, None], min=0, max=1)
    cloth0 = cloth_th[0].detach().cpu().numpy()
    return cloth0


# abandoned
def opendrDraw(vertCam0, face0, vertRgb0, winWidth, winHeight, focalLengthWidth, focalLengthHeight):
    assert len(vertCam0.shape) == 2 and vertCam0.shape[1] == 3
    assert len(face0.shape) == 2 and face0.shape[1] == 3
    nVert = vertCam0.shape[0]

    rn = ColoredRenderer()
    rn.camera = ProjectPoints(
        rt=np.zeros(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        f=np.array([focalLengthWidth, focalLengthHeight], dtype=np.float32),
        c=np.array([winWidth / 2., winHeight / 2.], dtype=np.float32),
        k=np.zeros(5, dtype=np.float32),
    )
    rn.frustum = {'near': 1.e-2, 'far': 1.e2, 'height': winHeight, 'width': winWidth}

    # set to rn
    rn.set(v=vertCam0, f=face0, vc=vertRgb0, bgcolor=np.ones((3,), dtype=np.float32))
    rn.background_image = np.ones((winHeight, winWidth, 3, ), dtype=np.float32)
    albedo = rn.vc
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=nVert,
        light_pos=np.array([186.60254038, -100., -123.20508076], dtype=np.float32) / 100.,
        vc=albedo,
        light_color=np.ones((3,), dtype=np.float32),
    )
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=nVert,
        light_pos=np.array([-659.80762114, 10., 542.82032303], dtype=np.float32) / 100.,
        vc=albedo,
        light_color=np.ones((3,), dtype=np.float32),
    )

    # fetch the image
    img0 = rn.r
    img0 = img0.astype(np.float32)

    return img0


# abandoned
# single view port visualization of mesh
def drawVertColoredMesh0_opendrBackend(backgroundImg0,
                                       focalLengthWidth, focalLengthHeight,
                                       E0, L0, U0,
                                       vert0, face0, vertRgb0):
    winWidth = backgroundImg0.shape[1]
    winHeight = backgroundImg0.shape[0]

    # Note E / L / U / vert0 should be in the same coord sys, world or cam
    assert len(E0) == 3
    assert len(L0) == 3
    assert len(U0) == 3
    assert len(vert0.shape) == 2 and vert0.shape[1] == 3
    assert len(face0.shape) == 2 and face0.shape[1] == 3
    nVert = vert0.shape[0]
    # tris0 = face0[:, [0, 2, 1]]
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, TexturedRenderer
    from opendr.lighting import LambertianPointLight
    rn = ColoredRenderer()
    rn.camera = ProjectPoints(
        rt=np.zeros(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        f=np.array([focalLengthWidth, focalLengthHeight], dtype=np.float32),
        c=np.array([winWidth / 2., winHeight / 2.], dtype=np.float32),
        k=np.zeros(5, dtype=np.float32),
    )
    rn.frustum = {'near': 1.e-2, 'far': 1.e2, 'height': winHeight, 'width': winWidth}

    # rotate everything
    cam0 = ELU02cam0(np.concatenate((E0, L0, U0), 0))
    R0 = cam0[:3, :3]
    Rtransposed0 = R0.transpose()
    T0 = cam0[:3, 3]
    vertNow0 = np.matmul(vert0, Rtransposed0) + T0[None]

    # set to rn
    rn.set(v=vertNow0, f=face0, vc=vertRgb0, bgcolor=np.ones((3, ), dtype=np.float32))
    rn.background_image = backgroundImg0
    albedo = rn.vc
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=nVert,
        light_pos=np.array([186.60254038, -100., -123.20508076], dtype=np.float32) / 100.,
        vc=albedo,
        light_color=np.ones((3, ), dtype=np.float32),
    )
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=nVert,
        light_pos=np.array([-659.80762114, 10., 542.82032303], dtype=np.float32) / 100.,
        vc=albedo,
        light_color=np.ones((3, ), dtype=np.float32),
    )

    # fetch the image
    img0 = rn.r
    img0 = img0.astype(np.float32)

    return img0


# abandoned
# generate a set of vis given different observation points
def drawVertColoredMesh0Multiview_opendrBackend(
        backgroundImg0,  # only the E=E0 will be the overlay background
        focalLengthWidth, focalLengthHeight,
        E0, L0, U0,  # Please make sure that E0's observation is the same as the backgrondImg (for overlay)
        vert0, face0, vertRgb0,
        EsOrderedDict):
    Es = OrderedDict([])
    Es['Original View Overlay'] = E0  # used for overlay
    Es.update(EsOrderedDict)  # will be drawn onto the white cloth

    rendering0s = OrderedDict([])
    for key in Es.keys():
        if key == 'Original View Overlay':
            rendering0 = drawVertColoredMesh0_opendrBackend(
                backgroundImg0=backgroundImg0,
                focalLengthWidth=focalLengthWidth, focalLengthHeight=focalLengthHeight,
                E0=Es[key], L0=L0, U0=U0,
                vert0=vert0, face0=face0, vertRgb0=vertRgb0,
            )
        else:
            rendering0 = drawVertColoredMesh0_opendrBackend(
                backgroundImg0=np.ones_like(backgroundImg0),
                focalLengthWidth=focalLengthWidth, focalLengthHeight=focalLengthHeight,
                E0=Es[key], L0=L0, U0=U0,
                vert0=vert0, face0=face0, vertRgb0=vertRgb0,
            )
        rendering0s[key] = rendering0
    return rendering0s


# just generate a set of observations points (Es) according to the given ELU
def pickInitialObservationPoints(E0, L0, ungrav0, dmax, numView):
    # You still need to postprocessing the result of this function by calling readjustObservationPoints

    # dmax = np.max(depthMap0[np.isfinite(depthMap0)])

    EL0_normalized = (L0 - E0) / np.linalg.norm(L0 - E0)
    ELNew0 = EL0_normalized * 0.42 * dmax
    LNew0 = E0 + ELNew0

    EPick0List = []

    for j in range(numView):
        LNewE0 = E0 - LNew0
        angleDegree = 360. * j / numView
        rotMat = getRotationMatrixBatchNP(
            ungrav0[None],
            np.array([angleDegree], dtype=np.float32)
        )[0]
        LNewEJ0 = np.dot(rotMat, LNewE0)

        EJ0 = LNew0 + LNewEJ0
        EPick0List.append(EJ0)

    LCenter0 = LNew0  # less general naming
    EPick0List = np.array(EPick0List)

    return EPick0List, LCenter0


# This function supposes U0 is on the same plain as ungrav0 and (L0-E0).
# CoReNet falls into this case (it looks down at the object, but without tilt / roll)
# If it has roll, (e.g. Scannet) then the produced UPick0List[0, :] would be different from U0
# which is likely not what you want.
def getUPick0List(EPick0List, LCenter0, ungrav0):
    # ungrav0 is the opposite direction of gravity
    # U0 is the up direction in the pixel map.
    # When you are not looking horizontally your U0 would not be the same as ungrav0
    delta = LCenter0[None, :] - EPick0List
    intermediate = np.cross(delta, ungrav0)
    UPick0List = np.cross(intermediate, delta)
    return UPick0List


# readjust the observation points (Es) according to the given vert0 and face0
# Avoid using this function if your U0 is different from ungrav0
def readjustObservationPoints(E0, L0, U0, EPick0List, LCenter0, vert0, face0, cudaDevice):
    # Note vert0 should be in the same sys as E0 / Epick0List !!!

    faceclockwise0 = face0[:, [0, 2, 1]]
    EPickReadjusted0List = []
    for j in range(len(EPick0List)):
        EPick0 = EPick0List[j].copy()
        flagContinue = True
        stepCount = 0
        LCenterEPick0_normalized = (-LCenter0 + EPick0) / np.linalg.norm(LCenter0 - EPick0)
        while flagContinue > 0:
            cam0 = ELU02cam0(np.concatenate((EPick0, LCenter0, U0), 0))
            R0 = cam0[:3, :3]
            Rtransposed0 = R0.transpose()
            T0 = cam0[:3, 3]
            vertNow0 = np.matmul(vert0, Rtransposed0) + T0[None]
            bufferTmp, _, _ = zBufferBatchNP(
                vertNow0[None], face0[None], cudaDevice=cudaDevice, height=128,
            )
            buffer0 = bufferTmp[0]
            bufferTmp_insideout, _, _ = zBufferBatchNP(
                vertNow0[None], faceclockwise0[None], cudaDevice=cudaDevice, height=128,
            )
            buffer0_insideout = bufferTmp_insideout[0]
            ifEPickOccupied = (buffer0 - buffer0_insideout > 0).sum() > 0.5 * np.prod(buffer0.shape)

            if stepCount < 20 and ifEPickOccupied:
                EPick0 = EPick0 - 0.1 * LCenterEPick0_normalized
            else:
                flagContinue = False
                # EPick0 = EPick0 - 0.3 * LCenterEPick0_normalized

            stepCount += 1

        EPickReadjusted0List.append(EPick0)

    return EPickReadjusted0List


# elevate to get the second half
def elevateObservationPoints(EPick0List_input, LCenter0, ungrav0):
    EPick0List = copy.deepcopy(EPick0List_input)
    for j in range(len(EPick0List)):
        x = EPick0List[j]
        # y = 1.3 * x - 0.3 * LCenter0 + 2. * U0
        y = x + 1.5 * ungrav0
        EPick0List.append(y)
    return EPick0List
