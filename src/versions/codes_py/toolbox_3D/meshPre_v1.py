# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import signed_distance_function_v1.cuda.mesh_preprocessing as meshPre_cuda


_LARGEFLOAT = 1.e10


def faceVertFindNextTHGPU(face, validFaceFlag):
    assert 'int32' in str(face.dtype)
    assert 'int32' in str(validFaceFlag.dtype)
    nextFaceVertLoc = torch.from_numpy(-2 * np.ones((face.shape[0], face.shape[1], 3, 2), dtype=np.int32)).cuda()
    nextFaceVertLoc = meshPre_cuda.faceVert_find_next(face, validFaceFlag, nextFaceVertLoc)
    assert nextFaceVertLoc.min() > -2
    return nextFaceVertLoc


def faceVertFindNextNP(face, validFaceFlag):
    nextFaceVertLoc_thgpu = faceVertFindNextTHGPU(
        torch.from_numpy(face).cuda(),
        torch.from_numpy(validFaceFlag).cuda(),
    )
    nextFaceVertLoc = nextFaceVertLoc_thgpu.detach().cpu().numpy()
    return nextFaceVertLoc


def faceEdgeFindNextTHGPU(face, validFaceFlag, nextFaceVertLoc, sumFaceEdgeLoc):
    assert 'int32' in str(face.dtype)
    assert 'int32' in str(validFaceFlag.dtype)
    assert 'int32' in str(nextFaceVertLoc.dtype)
    assert 'int32' in str(sumFaceEdgeLoc.dtype)
    nextFaceEdgeLoc = -2 * torch.ones_like(nextFaceVertLoc)
    nextFaceEdgeLoc = meshPre_cuda.faceEdge_find_next(face,
                                                      validFaceFlag,
                                                      nextFaceVertLoc,
                                                      sumFaceEdgeLoc,
                                                      nextFaceEdgeLoc)
    assert nextFaceEdgeLoc.min() > -2
    return nextFaceEdgeLoc


def faceEdgeFindNextNP(face, validFaceFlag, nextFaceVertLoc, sumFaceEdgeLoc):
    nextFaceEdgeLoc_thgpu = faceEdgeFindNextTHGPU(
        torch.from_numpy(face).cuda(),
        torch.from_numpy(validFaceFlag).cuda(),
        torch.from_numpy(nextFaceVertLoc).cuda(),
        torch.from_numpy(sumFaceEdgeLoc).cuda()
    )
    nextFaceEdgeLoc = nextFaceEdgeLoc_thgpu.detach().cpu().numpy()
    return nextFaceEdgeLoc


def countCycleFaceIndexingTHGPU(nextFaceIndexingLoc, validFaceFlag):
    assert 'int32' in str(nextFaceIndexingLoc.dtype)
    assert 'int32' in str(validFaceFlag.dtype)

    sumFaceIndexingLoc = torch.from_numpy(-2 * np.ones((nextFaceIndexingLoc.shape[0], nextFaceIndexingLoc.shape[1], 3), dtype=np.int32)).cuda()
    sumFaceIndexingLoc = meshPre_cuda.count_cycle_face_indexing(nextFaceIndexingLoc,
                                                                validFaceFlag,
                                                                sumFaceIndexingLoc)
    assert sumFaceIndexingLoc.min() > -2
    return sumFaceIndexingLoc

def countCycleFaceIndexingNP(nextFaceIndexingLoc, validFaceFlag):
    sumFaceIndexingLoc_thgpu = countCycleFaceIndexingTHGPU(
        torch.from_numpy(nextFaceIndexingLoc).cuda(),
        torch.from_numpy(validFaceFlag).cuda(),
    )
    sumFaceIndexingLoc = sumFaceIndexingLoc_thgpu.detach().cpu().numpy()
    return sumFaceIndexingLoc


def calcFaceEdgeNormalTHGPU(faceNormal, validFaceFlag, nextFaceEdgeLoc, sumFaceEdge):
    assert 'float' in str(faceNormal.dtype)
    assert 'int32' in str(validFaceFlag.dtype)
    assert 'int32' in str(nextFaceEdgeLoc.dtype)
    assert 'int32' in str(sumFaceEdge)

    faceEdgeNormal = torch.from_numpy(-2 * _LARGEFLOAT * np.ones((faceNormal.shape[0], faceNormal.shape[1], 3, 3), dtype=np.float32)).cuda()
    faceEdgeNormal = meshPre_cuda.calc_face_edge_normal(faceNormal,
                                                        validFaceFlag,
                                                        nextFaceEdgeLoc,
                                                        sumFaceEdge,
                                                        faceEdgeNormal)
    # So we are removing the following check, as there is likely that the mesh is problematic
    # that there are 4 faces connected to an edge, and then the norm of the normal becomes 0.
    # You should instead filter out these triangles.
    # assert faceEdgeNormal.min() > -1.01 * _LARGEFLOAT

    return faceEdgeNormal


def calcFaceEdgeNormalNP(faceNormal, validFaceFlag, nextFaceEdgeLoc, sumFaceEdge):
    faceEdgeNormal_thgpu = calcFaceEdgeNormalTHGPU(
        torch.from_numpy(faceNormal).cuda(),
        torch.from_numpy(validFaceFlag).cuda(),
        torch.from_numpy(nextFaceEdgeLoc).cuda(),
        torch.from_numpy(sumFaceEdge).cuda()
    )
    faceEdgeNormal = faceEdgeNormal_thgpu.detach().cpu().numpy()
    return faceEdgeNormal
