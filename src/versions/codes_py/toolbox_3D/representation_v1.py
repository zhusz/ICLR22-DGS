import torch
import math
from torch.nn.functional import grid_sample
from matplotlib.pyplot import get_cmap
from libmcubes_v0 import mcubes
import skimage.measure
from .mesh_surgery_v1 import combineMultiShapes_withVertRgb
import numpy as np


# various representations

# any line below can fully represent / determine a 3D object / scene
# but for any function call below, you do not need to input the whole (e.g. if you only wish to change "Any")

# vox based
#   Structure: (voxAny, goxyz, sCell)  - "vox" always means world sys - Any can be (Sdf / SdfSign)
#   Data Type Defined: (raycubeAny / camcubeAny / roomcubeAny, focalLengthXY)
#                                - "raycube" or "camcube" always means cam sys (goxyz is cam sys)
#                                - "roomcube" always mean world sys (goxyz is world sys)
#                                - Any can be (Sdf / SdfSign)
#                                - goxyz is not needed as your (0, 0, 0) is the camera location
#                                - sCell is not needed as your focalLengthXY and the "pixel" unit will do the job

# mesh based
#   (vert[World|Cam]0, face0/tris0)


def raycube2camcube(raycube, grid_sample_mode):
    assert len(raycube.shape) == 3
    assert 'float' in str(raycube.dtype)

    ny, nx, nz = raycube.shape
    lCamcube = []
    lMaskCamcube = []
    for jz in range(nz):
        factor = float(jz + 1) / nz

        xi = np.linspace(-1. / factor, 1. / factor, nx).astype(np.float32)
        yi = np.linspace(-1. / factor, 1. / factor, ny).astype(np.float32)
        x, y = np.meshgrid(xi, yi)  # 2D case, np.meshgrid works in the (x=col, y=row) way. 3D case, np.meshgrid works in the (y=row, x=col, z=depth way).
        # No matter which case, y is always row, and x is always col

        result = grid_sample(
            input=torch.from_numpy(raycube[:, :, jz][None, None, :, :]).cuda(),  # (1(bs), 1(c), nx(width), ny(yi)(height)
            grid=torch.from_numpy(np.stack([x.reshape(-1), y.reshape(-1)], -1)[None, :, None, :]).cuda(),  # (1(bs), len_x_y(qX), 1(qY), 2(xy))  # x is the row index!
            mode=grid_sample_mode,
            padding_mode='zeros',
            align_corners=False,
        )[0, 0, :, 0].detach().cpu().numpy().reshape((ny, nx))

        lCamcube.append(result)
        lMaskCamcube.append((np.abs(x) <= 1) & (np.abs(y) <= 1))

    camcube = np.stack(lCamcube, 2)
    maskCamcube = np.stack(lMaskCamcube, 2)

    return camcube, maskCamcube


def getRaycube2CamcubeNpone(nx, ny, nz):
    xNpone = np.zeros((ny, nx, nz), dtype=np.float32)
    yNpone = np.zeros((ny, nx, nz), dtype=np.float32)
    zNpone = np.zeros((ny, nx, nz), dtype=np.float32)
    for jz in range(nz):
        factor = float(jz + 1) / nz
        xi = np.linspace(-1. / factor, 1. / factor, nx).astype(np.float32)
        yi = np.linspace(-1. / factor, 1. / factor, ny).astype(np.float32)
        x, y = np.meshgrid(xi, yi)
        xNpone[:, :, jz] = x
        yNpone[:, :, jz] = y
        zNpone[:, :, jz] = float(jz + 1) / nz * 2. - 1.
    return xNpone, yNpone, zNpone


def raycube2camcube_batched_th_differentiable(raycube_th, grid_sample_mode):  # can be used by pytorch module
    assert len(raycube_th.shape) == 5  # (B, c, ny, nx, nz)
    assert 'float' in str(raycube_th.dtype)

    bsz, c, ny, nx, nz = raycube_th.shape
    xNpone, yNpone, zNpone = getRaycube2CamcubeNpone(nx, ny, nz)
    grid0 = np.stack([xNpone, yNpone, zNpone], 3)  # (ny, nx, nz, 3(xyz))
    grid = np.tile(grid0, (bsz, 1, 1, 1, 1))  # (bsz, ny, nx, nz, 3(xyz))
    grid_th = torch.from_numpy(grid).to(device=raycube_th.device)

    # camcube_th
    camcube_th = torch.stack([grid_sample(
        input=raycube_th[:, :, :, :, jz],  # (B, c, ny, nx)
        grid=grid_th[:, :, :, jz, :2],  # (B, ny, nx, 2(xy))
        mode=grid_sample_mode,
        padding_mode='zeros',
        align_corners=False,
    ) for jz in range(nz)], 4)  # (B, c, ny, nx, nz)

    # mask00_th
    mask00_np = ((np.abs(xNpone) <= 1) & (np.abs(yNpone) <= 1)).astype(np.float32)
    mask_np = np.tile(mask00_np, (bsz, c, 1, 1, 1))
    mask_th = torch.from_numpy(mask_np).to(camcube_th.device)

    return camcube_th, mask_th


def voxSdfSign2mesh_mc(voxSdfSign, goxyz, sCell):
    assert len(voxSdfSign.shape) == 3
    assert len(goxyz) == 3
    if type(sCell) is float:
        sCellX = sCell
        sCellY = sCell
        sCellZ = sCell
    else:
        assert len(sCell) == 3
        sCellX = float(sCell[0])
        sCellY = float(sCell[1])
        sCellZ = float(sCell[2])

    vert0, faceClockWise0 = mcubes.marching_cubes(voxSdfSign.transpose((1, 0, 2)), 0.5)  # since voxSdfSign == 1 is unoccupied
    face0 = faceClockWise0[:, [0, 2, 1]]
    vert0 = vert0.astype(np.float32)
    face0 = face0.astype(np.int32)

    vert0[:, 0] = (vert0[:, 0] - 0.5) * sCellX + float(goxyz[0])
    vert0[:, 1] = (vert0[:, 1] - 0.5) * sCellY + float(goxyz[1])
    vert0[:, 2] = (vert0[:, 2] - 0.5) * sCellZ + float(goxyz[2])

    return vert0, face0


def voxSdfSign2mesh_skmc(voxSdfSign, goxyz, sCell):
    assert len(voxSdfSign.shape) == 3
    assert len(goxyz) == 3
    if type(sCell) is float:
        sCellX = sCell
        sCellY = sCell
        sCellZ = sCell
    else:
        assert len(sCell) == 3
        sCellX = float(sCell[0])
        sCellY = float(sCell[1])
        sCellZ = float(sCell[2])

    # vert0, faceClockWise0 = mcubes.marching_cubes(voxSdfSign.transpose((1, 0, 2)), 0.5)  # since voxSdfSign == 1 is unoccupied
    vert0, face0, normal0, value0 = skimage.measure.marching_cubes(
        voxSdfSign.transpose((1, 0, 2)), level=0.5, spacing=[1., 1., 1.], method='lewiner',
    )

    # face0 = faceClockWise0[:, [0, 2, 1]]  # It seems this function does not need to do wrapping.
    # It has one sign gap.
    vert0 = vert0.astype(np.float32)
    face0 = face0.astype(np.int32)

    vert0[:, 0] = (vert0[:, 0] - 0.5) * sCellX + float(goxyz[0])
    vert0[:, 1] = (vert0[:, 1] - 0.5) * sCellY + float(goxyz[1])
    vert0[:, 2] = (vert0[:, 2] - 0.5) * sCellZ + float(goxyz[2])

    return vert0, face0


def voxSdf2mesh_skmc(voxSdf, goxyz, sCell):
    assert len(voxSdf.shape) == 3
    assert len(goxyz) == 3
    if type(sCell) is float:
        sCellX = sCell
        sCellY = sCell
        sCellZ = sCell
    else:
        assert len(sCell) == 3
        sCellX = float(sCell[0])
        sCellY = float(sCell[1])
        sCellZ = float(sCell[2])

    # vert0, faceClockWise0 = mcubes.marching_cubes(voxSdfSign.transpose((1, 0, 2)), 0.5)  # since voxSdfSign == 1 is unoccupied
    vert0, face0, normal0, value0 = skimage.measure.marching_cubes(
        voxSdf.transpose((1, 0, 2)), level=0.0, spacing=[1., 1., 1.], method='lewiner',
    )

    # face0 = faceClockWise0[:, [0, 2, 1]]  # It seems this function does not need to do wrapping.
    # It has one sign gap.
    vert0 = vert0.astype(np.float32)
    face0 = face0.astype(np.int32)

    vert0[:, 0] = (vert0[:, 0] - 0.5) * sCellX + float(goxyz[0])
    vert0[:, 1] = (vert0[:, 1] - 0.5) * sCellY + float(goxyz[1])
    vert0[:, 2] = (vert0[:, 2] - 0.5) * sCellZ + float(goxyz[2])

    return vert0, face0


# Note this function is for voxlabeling with multiple colors. For the binary case (occupied / non-occupied) please directly call pytorch3d.ops.cubify()
# We do not do on 0 (which is air)
def voxYXZLabeling2mesh_twelveTriangles(voxLabeling, goxyz, sCell, colorPool, device):
    from pytorch3d.ops import cubify
    # We assumes that labeling means: 0: air, positive values: categories occupied
    assert len(voxLabeling.shape) == 3
    assert len(goxyz) == 3
    if type(sCell) is float:
        sCellX = sCell
        sCellY = sCell
        sCellZ = sCell
    else:
        assert len(sCell) == 3
        sCellX = float(sCell[0])
        sCellY = float(sCell[1])
        sCellZ = float(sCell[2])

    # cmap = get_cmap('spring')
    # colorPool = np.stack([
    #     cmap(float(k) / colorTot)[:3] for k in range(colorTot + 1)
    # ], 0)
    voxLabelingZYX = voxLabeling.transpose((2, 0, 1))  # yxz to zyx
    if type(voxLabeling) is torch.Tensor:
        voxLabelingZYX_thDevice = voxLabelingZYX.detach().clone().to(device)
    else:
        voxLabelingZYX_thDevice = torch.from_numpy(voxLabelingZYX.copy()).to(device)
    uniqueLabeling_thDevice = torch.unique(voxLabelingZYX_thDevice)
    assert uniqueLabeling_thDevice[0].item() == 0

    if len(uniqueLabeling_thDevice) > 1:
        voxOneHot_thDevice = torch.stack([
            (voxLabelingZYX_thDevice == uniqueLabeling_thDevice[k].item()).float() for k in range(1, len(uniqueLabeling_thDevice))
        ], 0)  # Note no zero
        meshes = cubify(voxels=voxOneHot_thDevice, thresh=0.5, device=device)

        # Combined Total Mesh
        vert_face_faceColor_list = [
            (meshes.verts_list()[k - 1].cpu().numpy(),
             meshes.faces_list()[k - 1].cpu().numpy(),
             np.tile(colorPool[uniqueLabeling_thDevice[k].item()][None, :], (meshes.faces_list()[k - 1].shape[0], 1)))
            for k in range(1, len(uniqueLabeling_thDevice))
        ]
        ttVert0, ttFace0, ttFaceColor0 = combineMultiShapes_withVertRgb(vert_face_faceColor_list)
        # We now have our def of "face" following conventional, so comment out the following line
        # ttFace0 = ttFace0[:, [0, 2, 1]]  # our def of "face"

        goXYZ = goxyz + np.array(voxLabeling.shape, dtype=np.float32)[[1, 0, 2]] * np.array([sCellX, sCellY, sCellZ], dtype=np.float32)

        ttVert0 = ((ttVert0 + 1.) / 2.) * (goXYZ - goxyz)[None, :] + goxyz[None, :]

        # Each Mesh
        semanticMeshRetrieveList = {}
        lenUniqueNonAirSemanticCate = uniqueLabeling_thDevice.shape[0] - 1
        for j in range(lenUniqueNonAirSemanticCate):
            c = int(uniqueLabeling_thDevice[j + 1])
            semanticMeshRetrieveList[c] = {}
            semanticMeshRetrieveList[c]['vert'] = \
                ((meshes.verts_list()[j].detach().cpu().float().numpy() + 1.) / 2.) * \
                (goXYZ - goxyz)[None, :] + goxyz[None, :]
            semanticMeshRetrieveList[c]['face'] = meshes.faces_list()[j].detach().cpu().int().numpy()
    else:
        ttVert0 = np.eye(3).astype(np.float32)
        ttFace0 = np.array([[0, 1, 2]], dtype=np.int32)
        ttFaceColor0 = np.zeros((1, 3), dtype=np.float32)
        semanticMeshRetrieveList = {}

    return ttVert0, ttFace0, ttFaceColor0, semanticMeshRetrieveList


def minMaxBound2vox_yxz(minBound, maxBound, supposedNVox):
    assert minBound.shape == (3, )
    assert maxBound.shape == (3, )

    assert np.all(maxBound - minBound > 0.)
    goxyz = minBound.copy()
    sCell = (np.prod(maxBound - minBound) / supposedNVox) ** (1. / 3.)
    sCell = np.array([sCell, sCell, sCell], dtype=np.float32)
    Ls = np.ceil((maxBound - minBound) / sCell).astype(np.int32)
    Lx = Ls[0]
    Ly = Ls[1]
    Lz = Ls[2]
    xi = np.linspace(goxyz[0], goxyz[0] + (Lx - 1) * sCell[0], Lx).astype(np.float32)
    yi = np.linspace(goxyz[1], goxyz[1] + (Ly - 1) * sCell[1], Ly).astype(np.float32)
    zi = np.linspace(goxyz[2], goxyz[2] + (Lz - 1) * sCell[2], Lz).astype(np.float32)
    x, y, z = np.meshgrid(xi, yi, zi)  # Note, now the cube storage (like x, y, z here or cubeSdf etc) will be indexed in the a[y, x, z] fashion! This is also why the shape of voxXyz below is (Ly, Lx, Lz, 3) instead of (Lx, Ly, Lz, 3)
    voxXyz = np.stack([x, y, z], 3)  # (Ly, Lx, Lz, 3)
    return voxXyz, goxyz, sCell


def directDepthMap2PCNP(depthMap0, fScaleWidth, fScaleHeight):
    assert len(depthMap0.shape) == 2
    assert np.nanmin(depthMap0) > 0
    width = int(depthMap0.shape[1])
    height = int(depthMap0.shape[0])
    uBorder = 1. - 1. / width
    u = np.linspace(-uBorder, uBorder, width).astype(np.float32)
    vBorder = 1. - 1. / height
    v = np.linspace(-vBorder, vBorder, height).astype(np.float32)
    u, v = np.meshgrid(u, v)
    maskFinite = np.isfinite(depthMap0)
    xyzCamPersp0 = np.stack([u[maskFinite], v[maskFinite], depthMap0[maskFinite]], 1)
    xyzCam0 = np.stack([
        xyzCamPersp0[:, 0] * xyzCamPersp0[:, 2] / fScaleWidth,
        xyzCamPersp0[:, 1] * xyzCamPersp0[:, 2] / fScaleHeight,
        xyzCamPersp0[:, 2],
    ], 1)
    return xyzCam0, xyzCamPersp0

