# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .mesh_v1 import vertInfo2faceVertInfoNP
from scipy.interpolate import interp2d, RectBivariateSpline
import torch
import signed_distance_function_v1.cuda.sdf_core as sdf_cuda
import math
import numpy as np
import cv2
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def mesh0ToRearrange0NPV1(vertFace0, roughTotalGrids):
    # Input: vertFace0: nTris * 3(012) * 3(xyz)
    # Outputs: try it yourself

    assert len(vertFace0.shape) == 3
    nTris = vertFace0.shape[0]
    assert vertFace0.shape[1] == 3
    assert vertFace0.shape[2] == 3
    assert 'float' in str(vertFace0.dtype)
    assert type(roughTotalGrids) is int

    xmin = vertFace0[:, :, 0].min()
    ymin = vertFace0[:, :, 1].min()
    zmin = vertFace0[:, :, 2].min()
    xmax = vertFace0[:, :, 0].max()
    ymax = vertFace0[:, :, 1].max()
    zmax = vertFace0[:, :, 2].max()

    '''  This is only introduced in Version 2 (A13 series)
    slack = 0.05
    xmin, xmax = (1. + slack) * xmin + (-slack) * xmax, (-slack) * xmin + (1. + slack) * xmax
    ymin, ymax = (1. + slack) * ymin + (-slack) * ymax, (-slack) * ymin + (1. + slack) * ymax
    zmin, zmax = (1. + slack) * zmin + (-slack) * zmax, (-slack) * zmin + (1. + slack) * zmax
    '''

    v = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    vGrid = v / roughTotalGrids
    aGrid = vGrid ** (1. / 3.)
    lenGridX = int(math.ceil((xmax - xmin) / aGrid))
    lenGridY = int(math.ceil((ymax - ymin) / aGrid))
    lenGridZ = int(math.ceil((zmax - zmin) / aGrid))
    totalGrid = lenGridX * lenGridY * lenGridZ

    vertFaceCentroid0 = vertFace0.mean(1)
    vertFaceGridXYZ0 = -np.ones((nTris, 3), dtype=np.int32)
    vertFaceGridXYZ0[:, 0] = (np.floor((vertFaceCentroid0[:, 0] - xmin) / aGrid)).astype(np.int32)
    vertFaceGridXYZ0[:, 1] = (np.floor((vertFaceCentroid0[:, 1] - ymin) / aGrid)).astype(np.int32)
    vertFaceGridXYZ0[:, 2] = (np.floor((vertFaceCentroid0[:, 2] - zmin) / aGrid)).astype(np.int32)

    vertFaceGridXYZ_thcpu = torch.from_numpy(vertFaceGridXYZ0[None])
    trisIDOrderingBuffer_thcpu = torch.from_numpy(-np.ones((1, nTris), dtype=np.int32))
    gridTrisStartingIndex_thcpu = torch.from_numpy(-np.ones((1, lenGridX, lenGridY, lenGridZ), dtype=np.int32))
    trisIDOrderingBuffer_thcpu, gridTrisStartingIndex_thcpu = sdf_cuda.point_id_rearrange(
        vertFaceGridXYZ_thcpu, trisIDOrderingBuffer_thcpu, gridTrisStartingIndex_thcpu
    )
    trisIDOrderingBuffer0 = trisIDOrderingBuffer_thcpu[0].numpy()
    gridTrisStartingIndex0 = gridTrisStartingIndex_thcpu[0].numpy()

    # Assert the triangle area
    vertFaceXMin0 = vertFace0[:, :, 0].min(1)
    vertFaceXMax0 = vertFace0[:, :, 0].max(1)
    vertFaceYMin0 = vertFace0[:, :, 1].min(1)
    vertFaceYMax0 = vertFace0[:, :, 1].max(1)
    vertFaceZMin0 = vertFace0[:, :, 2].min(1)
    vertFaceZMax0 = vertFace0[:, :, 2].max(1)
    vertFaceGridXMin0 = (np.floor((vertFaceXMin0 - xmin) / aGrid)).astype(np.int32)
    vertFaceGridXMax0 = (np.floor((vertFaceXMax0 - xmin) / aGrid)).astype(np.int32)
    vertFaceGridYMin0 = (np.floor((vertFaceYMin0 - ymin) / aGrid)).astype(np.int32)
    vertFaceGridYMax0 = (np.floor((vertFaceYMax0 - ymin) / aGrid)).astype(np.int32)
    vertFaceGridZMin0 = (np.floor((vertFaceZMin0 - zmin) / aGrid)).astype(np.int32)
    vertFaceGridZMax0 = (np.floor((vertFaceZMax0 - zmin) / aGrid)).astype(np.int32)
    vertFaceGridVol0 = (vertFaceGridXMax0 - vertFaceGridXMin0 + 1) * \
                   (vertFaceGridYMax0 - vertFaceGridYMin0 + 1) * \
                   (vertFaceGridZMax0 - vertFaceGridZMin0 + 1)

    meshGridMetaDict0 = {
        'nFace': nTris,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'zmin': zmin,
        'zmax': zmax,
        'aGrid': aGrid,
        'lenGridX': lenGridX,
        'lenGridY': lenGridY,
        'lenGridZ': lenGridZ,
        'totalGrid': totalGrid,
    }

    return trisIDOrderingBuffer0, gridTrisStartingIndex0, vertFaceGridVol0, meshGridMetaDict0, \
           np.stack([vertFaceGridXMin0, vertFaceGridYMin0, vertFaceGridZMin0,
                     vertFaceGridXMax0, vertFaceGridYMax0, vertFaceGridZMax0], 1)


def computeSDFTHGPU(queryPoints, trisIDOrderingBuffer, gridTrisStartingIndex, faceVert,
                    faceNormal, edgeNormal, faceVertNormal, validFaceFlag,
                    meshGridMetaDict0, debuggingInfoExportDim=0, ifCheckNorm=True):

    # Inputs:
    #   queryPoints: m * nQuery * 3(xyz)
    #   trisIDOrderingBuffer: m * nTris
    #   gridTrisStartingIndex: m * totalGrid (remember to resize)
    #   faceVert: m * nTris * 3(012) * 3(xyz)
    #   faceNormal: m * nTris * 3(nxnynz)
    #   edgeNormal: m * nTris * 3(122001) * 3(nxnynz)
    #   faceVertNormal: m * nTris * 3(012) * 3(nxnynz)
    #   validFaceFlag(int32): m * nFace
    #   meshGridMetaDict0: must contain [x/y/z][min/max] and aGrid (the a of one grid)

    # Outputs:
    #   sdf: m * nQuery // positive is outside, and negative is inside
    #   projPoint: m * nQuery * 3
    #   projTrisID: m * nQuery
    #   debuggingInfoExport: m * nQuery * dimDebug

    # check all the input normals to be normalized!

    assert 'int32' in str(validFaceFlag.dtype)
    m = queryPoints.shape[0]
    nQuery = queryPoints.shape[1]

    assert (torch.isnan(queryPoints) == 0).all()
    assert (torch.isinf(queryPoints) == 0).all()

    if ifCheckNorm:
        assert abs(torch.norm(faceNormal[validFaceFlag > 0], p=2, dim=-1) - 1.).max() < 1.e-3
        assert abs(torch.norm(edgeNormal[validFaceFlag > 0], p=2, dim=-1) - 1.).max() < 1.e-3
        assert abs(torch.norm(faceVertNormal[validFaceFlag > 0], p=2, dim=-1) - 1.).max() < 1.e-3

    sdfPos = torch.from_numpy(-2 * np.ones((m, nQuery), dtype=np.float32)).cuda()
    sdfNeg = torch.from_numpy(-2 * np.ones((m, nQuery), dtype=np.float32)).cuda()
    projPointPos = torch.from_numpy(-2 * np.ones((m, nQuery, 3), dtype=np.float32)).cuda()
    projPointNeg = torch.from_numpy(-2 * np.ones((m, nQuery, 3), dtype=np.float32)).cuda()
    projPointNormalPos = torch.from_numpy(-2 * np.ones((m, nQuery, 3), dtype=np.float32)).cuda()
    projPointNormalNeg = torch.from_numpy(-2 * np.ones((m, nQuery, 3), dtype=np.float32)).cuda()
    projTrisIDPos = torch.from_numpy(-2 * np.ones((m, nQuery), dtype=np.int32)).cuda()
    projTrisIDNeg = torch.from_numpy(-2 * np.ones((m, nQuery), dtype=np.int32)).cuda()
    debuggingInfoExport = torch.from_numpy(-2 * np.ones((m, nQuery, debuggingInfoExportDim), dtype=np.float32)).cuda()

    sdfPos, sdfNeg, projPointPos, projPointNeg, projPointNormalPos, projPointNormalNeg, projTrisIDPos, projTrisIDNeg, debuggingInfoExport = sdf_cuda.sdf_tris_search(
        queryPoints, trisIDOrderingBuffer, gridTrisStartingIndex, faceVert,
        faceNormal, edgeNormal, faceVertNormal, validFaceFlag,
        sdfPos, sdfNeg, projPointPos, projPointNeg, projPointNormalPos, projPointNormalNeg, projTrisIDPos,
        projTrisIDNeg,
        debuggingInfoExport,
        meshGridMetaDict0['xmin'], meshGridMetaDict0['xmax'],
        meshGridMetaDict0['ymin'], meshGridMetaDict0['ymax'],
        meshGridMetaDict0['zmin'], meshGridMetaDict0['zmax'],
        meshGridMetaDict0['aGrid']
    )

    # assert ((sdfPos < 1e8) | (sdfNeg < 1e8)).all()

    return sdfPos, sdfNeg, projPointPos, projPointNeg, projPointNormalPos, projPointNormalNeg, projTrisIDPos, projTrisIDNeg, debuggingInfoExport


def computeSDFNP(queryPoints, trisIDOrderingBuffer, gridTrisStartingIndex, faceVert,
                 faceNormal, edgeNormal, faceVertNormal, validFaceFlag,
                 meshGridMetaDict0, debuggingInfoExportDim=0, ifCheckNorm=True):
    sdfPos_thgpu, sdfNeg_thgpu, projPointPos_thgpu, projPointNeg_thgpu, projPointNormalPos_thgpu, projPointNormalNeg_thgpu, projTrisIDPos_thgpu, projTrisIDNeg_thgpu, debuggingInfoExport_thgpu = computeSDFTHGPU(
        torch.from_numpy(queryPoints).cuda(),
        torch.from_numpy(trisIDOrderingBuffer).cuda(),
        torch.from_numpy(gridTrisStartingIndex).cuda(),
        torch.from_numpy(faceVert).cuda(),
        torch.from_numpy(faceNormal).cuda(),
        torch.from_numpy(edgeNormal).cuda(),
        torch.from_numpy(faceVertNormal).cuda(),
        torch.from_numpy(validFaceFlag).cuda(),
        meshGridMetaDict0,
        debuggingInfoExportDim=debuggingInfoExportDim,
        ifCheckNorm=ifCheckNorm
    )
    sdfPos = sdfPos_thgpu.detach().cpu().numpy()
    sdfNeg = sdfNeg_thgpu.detach().cpu().numpy()
    projPointPos = projPointPos_thgpu.detach().cpu().numpy()
    projPointNeg = projPointNeg_thgpu.detach().cpu().numpy()
    projPointNormalPos = projPointNormalPos_thgpu.detach().cpu().numpy()
    projPointNormalNeg = projPointNormalNeg_thgpu.detach().cpu().numpy()
    projTrisIDPos = projTrisIDPos_thgpu.detach().cpu().numpy()
    projTrisIDNeg = projTrisIDNeg_thgpu.detach().cpu().numpy()
    debuggingInfoExport = debuggingInfoExport_thgpu.detach().cpu().numpy()
    return sdfPos, sdfNeg, projPointPos, projPointNeg, projPointNormalPos, projPointNormalNeg, projTrisIDPos, projTrisIDNeg, debuggingInfoExport
