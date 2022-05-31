# (tfconda)
from lib2to3.pgen2.pgen import NFAState
from scipy.misc import face
import time
import torch
from .mesh_v1 import vertInfo2faceVertInfoTHGPU, vertInfo2faceVertInfoNP
from ..toolbox_torch.general_algorithm_wrapper_v1 import batched_binary_search_over_cumsumed_vec_thgpu
import numpy as np
from ..np_ext.data_processing_utils_v1 import d2, determinedArbitraryPermutedVector2correctcorrect


def mesh_sampling(num_sampling, vert0_thgpu, face0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu,
                  ifReweightedByTriangleArea=False):
    assert len(vert0_thgpu.shape) == 2 and vert0_thgpu.shape[1] == 3
    nVert = vert0_thgpu.shape[0]
    assert len(face0_thgpu.shape) == 2 and face0_thgpu.shape[1] == 3
    nFace = face0_thgpu.shape[0]

    faceVert0_thgpu = vertInfo2faceVertInfoTHGPU(vert0_thgpu[None], face0_thgpu[None])[0]
    faceNormal0_thgpu = torch.cross(
        faceVert0_thgpu[:, 1, :] - faceVert0_thgpu[:, 0, :],
        faceVert0_thgpu[:, 2, :] - faceVert0_thgpu[:, 0, :],
    )
    if not ifReweightedByTriangleArea:
        return mesh_sampling_given_normal(
            num_sampling, faceVert0_thgpu, faceNormal0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu,
        )
    else:
        faceArea0_thgpu = torch.norm(faceNormal0_thgpu, dim=1)
        faceCumsumWeight0_thgpu = torch.cumsum(faceArea0_thgpu / faceArea0_thgpu.sum(), dim=0)
        return mesh_weighted_sampling_given_normal(
            num_sampling, faceCumsumWeight0_thgpu,
            faceVert0_thgpu, faceNormal0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu,
        )


def mesh_sampling_given_normal(
        num_sampling, faceVert0_thgpu, faceNormal0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu):
    assert len(faceVert0_thgpu.shape) == 3 and faceVert0_thgpu.shape[1:] == (3, 3)
    nFace = faceVert0_thgpu.shape[0]
    assert len(faceNormal0_thgpu.shape) == 2 and faceNormal0_thgpu.shape == (nFace, 3)

    if nFace == 0:
        point0_thgpu = torch.zeros(num_sampling, 3, dtype=torch.float32,
                                    device=faceVert0_thgpu.device)
        pointNormal0_thgpu = torch.zeros_like(point0_thgpu)
        pointVertInfo0_thgpu = {}
        for k in vertInfo0_thgpu.keys():
            pointVertInfo0_thgpu[k] = torch.zeros(
                num_sampling, vertInfo0_thgpu[k].shape[1],
                dtype=vertInfo0_thgpu[k].dtype, device=vertInfo0_thgpu[k].device,
            )
        pointFaceInfo0_thgpu = {}
        for k in faceInfo0_thgpu.keys():
            pointFaceInfo0_thgpu[k] = torch.zeros(
                num_sampling, faceInfo0_thgpu[k].shape[1],
                dtype=faceInfo0_thgpu[k].dtype, device=faceInfo0_thgpu[k].device,
            )

    else:
        # randomness
        pointFace0_thgpu = torch.randint(low=0, high=nFace, size=(num_sampling,),
                                         device=faceVert0_thgpu.device).long()
        p1 = torch.rand(num_sampling, device=faceVert0_thgpu.device).float()
        p2 = torch.rand(num_sampling, device=faceVert0_thgpu.device).float()
        pLeft = torch.min(p1, p2)
        pRight = torch.max(p1, p2)
        pointBary0_thgpu = torch.stack([pLeft, pRight - pLeft, 1. - pRight], 1)

        # cache
        pointFaceVert0_thgpu = faceVert0_thgpu[pointFace0_thgpu, :, :]
        pointFaceNormal0_thgpu = faceNormal0_thgpu[pointFace0_thgpu, :]
        pointFaceNormal0_thgpu = pointFaceNormal0_thgpu / \
                                 (torch.norm(pointFaceNormal0_thgpu, p=2, dim=1)[:, None] + 1.e-5)

        # point0_thgpu and pointNormal0_thgpu
        point0_thgpu = pointBary0_thgpu[:, 0, None] * pointFaceVert0_thgpu[:, 0, :] + \
                       pointBary0_thgpu[:, 1, None] * pointFaceVert0_thgpu[:, 1, :] + \
                       pointBary0_thgpu[:, 2, None] * pointFaceVert0_thgpu[:, 2, :]
        pointNormal0_thgpu = pointFaceNormal0_thgpu

        # pointVertInfo0_thgpu
        pointVertInfo0_thgpu = {}
        for k in vertInfo0_thgpu.keys():
            # assert len(vertInfo0_thgpu[k].shape) == 2 and vertInfo0_thgpu[k].shape[0] == nVert
            faceVertInfo0_thgpu = vertInfo2faceVertInfoTHGPU(vertInfo0_thgpu[k][None], face0_thgpu[None])[0]
            pointFaceVertInfo0_thgpu = faceVertInfo0_thgpu[pointFace0_thgpu]
            pointVertInfo0_thgpu[k] = pointBary0_thgpu[:, 0, None] * pointFaceVertInfo0_thgpu[:, 0, :] + \
                                      pointBary0_thgpu[:, 1, None] * pointFaceVertInfo0_thgpu[:, 1, :] + \
                                      pointBary0_thgpu[:, 2, None] * pointFaceVertInfo0_thgpu[:, 2, :]

        # pointFaceInfo0_thgpu
        pointFaceInfo0_thgpu = {}
        for k in faceInfo0_thgpu.keys():
            assert len(faceInfo0_thgpu[k].shape) == 2 and faceInfo0_thgpu[k].shape[0] == nFace
            pointFaceInfo0_thgpu[k] = faceInfo0_thgpu[k][pointFace0_thgpu, :]

    return point0_thgpu, pointNormal0_thgpu, pointVertInfo0_thgpu, pointFaceInfo0_thgpu


def mesh_weighted_sampling_given_normal(
        num_sampling, faceCumsumWeight0_thgpu,
        faceVert0_thgpu, faceNormal0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu):
    # Note the weight is per triangle, rather than per area.
    # If you want to uniformly sample from all the triangles, then larger triangles should
    # come with larger weight.
    assert len(faceVert0_thgpu.shape) == 3 and faceVert0_thgpu.shape[1:] == (3, 3)
    nFace = faceVert0_thgpu.shape[0]
    assert len(faceNormal0_thgpu.shape) == 2 and faceNormal0_thgpu.shape == (nFace, 3)
    assert len(faceCumsumWeight0_thgpu.shape) == 1
    assert faceCumsumWeight0_thgpu.shape[0] == nFace

    if faceCumsumWeight0_thgpu[-1] == 0:
        faceCumsumWeight0_thgpu = torch.from_numpy(
            np.arange(nFace).astype(np.float32) / nFace,
        ).to(faceCumsumWeight0_thgpu.device)

    # randomness
    q_thgpu = torch.rand(num_sampling, dtype=torch.float32, device=faceVert0_thgpu.device) \
              * faceCumsumWeight0_thgpu[-1]
    pointFace0_thgpu = batched_binary_search_over_cumsumed_vec_thgpu(
        values=q_thgpu, cumsumed_vec=faceCumsumWeight0_thgpu,
    ).long()
    pointFace0_thgpu[pointFace0_thgpu < 0] = 0
    pointFace0_thgpu[pointFace0_thgpu >= nFace] = nFace - 1
    p12_thgpu = torch.rand(num_sampling, device=faceVert0_thgpu.device).float()
    p3_thgpu = torch.rand(num_sampling, device=faceVert0_thgpu.device).float() ** 0.5
    pointBary0_thgpu = torch.stack([(1 - p12_thgpu) * p3_thgpu, p12_thgpu * p3_thgpu, 1 - p3_thgpu], 1)

    # cache
    pointFaceVert0_thgpu = faceVert0_thgpu[pointFace0_thgpu, :, :]
    pointFaceNormal0_thgpu = faceNormal0_thgpu[pointFace0_thgpu, :]
    pointFaceNormal0_thgpu = pointFaceNormal0_thgpu / \
                             (torch.norm(pointFaceNormal0_thgpu, p=2, dim=1)[:, None] + 1.e-5)

    # point0_thgpu and pointNormal0_thgpu
    point0_thgpu = pointBary0_thgpu[:, 0, None] * pointFaceVert0_thgpu[:, 0, :] + \
                   pointBary0_thgpu[:, 1, None] * pointFaceVert0_thgpu[:, 1, :] + \
                   pointBary0_thgpu[:, 2, None] * pointFaceVert0_thgpu[:, 2, :]
    pointNormal0_thgpu = pointFaceNormal0_thgpu

    # pointVertInfo0_thgpu
    pointVertInfo0_thgpu = {}
    for k in vertInfo0_thgpu.keys():
        # assert len(vertInfo0_thgpu[k].shape) == 2 and vertInfo0_thgpu[k].shape[0] == nVert
        faceVertInfo0_thgpu = vertInfo2faceVertInfoTHGPU(vertInfo0_thgpu[k][None], face0_thgpu[None])[0]
        pointFaceVertInfo0_thgpu = faceVertInfo0_thgpu[pointFace0_thgpu]
        pointVertInfo0_thgpu[k] = pointBary0_thgpu[:, 0, None] * pointFaceVertInfo0_thgpu[:, 0, :] + \
                                  pointBary0_thgpu[:, 1, None] * pointFaceVertInfo0_thgpu[:, 1, :] + \
                                  pointBary0_thgpu[:, 2, None] * pointFaceVertInfo0_thgpu[:, 2, :]

    # pointFaceInfo0_thgpu
    pointFaceInfo0_thgpu = {}
    for k in faceInfo0_thgpu.keys():
        assert len(faceInfo0_thgpu[k].shape) == 2 and faceInfo0_thgpu[k].shape[0] == nFace
        pointFaceInfo0_thgpu[k] = faceInfo0_thgpu[k][pointFace0_thgpu, :]

    return point0_thgpu, pointNormal0_thgpu, pointVertInfo0_thgpu, pointFaceInfo0_thgpu


def mesh_sampling_given_normal_np_simple(num_sampling, faceVert0):
    assert len(faceVert0.shape) == 3 and faceVert0.shape[1:] == (3, 3)
    nFace = faceVert0.shape[0]
    assert nFace > 0

    pointFace0 = np.random.randint(low=0, high=nFace, size=(num_sampling, ), dtype=np.int32)
    p1 = np.random.rand(num_sampling, ).astype(np.float32)
    p2 = np.random.rand(num_sampling, ).astype(np.float32)
    pLeft = np.minimum(p1, p2)
    pRight = np.maximum(p1, p2)
    pointBary0 = np.stack([pLeft, pRight - pLeft, 1. - pRight], 1)

    pointFaceVert0 = faceVert0[pointFace0, :, :]
    point0 = pointBary0[:, 0, None] * pointFaceVert0[:, 0, :] + \
             pointBary0[:, 1, None] * pointFaceVert0[:, 1, :] + \
             pointBary0[:, 2, None] * pointFaceVert0[:, 2, :]
    return {
        'point0': point0,
        'pointFace0': pointFace0,
    }


def mesh_weighted_sampling_given_normal_np(
        num_sampling, faceCumsumWeight0, faceVert0, faceNormal0, vertInfo0, faceInfo0, **kwargs):
    ifFaceNormal0alreadyNormalized = kwargs['ifFaceNormal0alreadyNormalized']
    cudaDevice = kwargs['cudaDevice']

    assert len(faceVert0.shape) == 3 and faceVert0.shape[1:] == (3, 3)
    nFace = int(faceVert0.shape[0])
    assert len(faceNormal0.shape) == 2 and faceNormal0.shape == (nFace, 3)
    assert len(faceCumsumWeight0.shape) == 1
    assert faceCumsumWeight0.shape[0] == nFace

    if faceCumsumWeight0[-1] == 0:
        faceCumsumWeight0 = np.arange(nFace).astype(np.float32)
    
    # randomness
    q = np.random.rand(num_sampling).astype(np.float32) * faceCumsumWeight0[-1]
    q[0] = 1.e-6 + faceCumsumWeight0[0]
    # pointFace0 = np.argmax(q[:, None] < faceCumsumWeight0[None, :], axis=1)
    pointFace0 = batched_binary_search_over_cumsumed_vec_thgpu(
        values=torch.from_numpy(q).float().to(cudaDevice),
        cumsumed_vec=torch.from_numpy(faceCumsumWeight0).float().to(cudaDevice),
    ).detach().cpu().numpy().astype(np.int32)
    pointFace0[pointFace0 < 0] = 0
    pointFace0[pointFace0 >= nFace] = nFace - 1
    p12 = np.random.rand(num_sampling).astype(np.float32)
    p3 = np.random.rand(num_sampling).astype(np.float32) ** 0.5
    pointBary0 = np.stack([(1 - p12) * p3, p12 * p3, 1 - p3], 1)
    
    if ifFaceNormal0alreadyNormalized:
        pass
    else:
        raise NotImplementedError('Not yet verified')
        faceNormal0 = faceNormal0 / \
            np.clip(np.linalg.norm(faceNormal0, ord=2, axis=1)[:, None], a_min=1.e-5, a_max=np.inf)

    pointFaceVert0 = faceVert0[pointFace0, :, :]
    pointFaceNormal0 = faceNormal0[pointFace0, :]
    point0 = (pointBary0[:, :, None] * pointFaceVert0).sum(1)
    pointNormal0 = pointFaceNormal0

    pointVertInfo0 = {}
    pointFaceInfo0 = {}
    if len(vertInfo0.keys()) > 0 or len(faceInfo0.keys()) > 0:
        raise NotImplementedError
    
    return point0, pointNormal0, pointFace0, pointVertInfo0, pointFaceInfo0


def mesh_weighted_sampling_given_normal_fixed_rand(
        num_sampling, faceCumsumWeight0_thgpu,
        faceVert0_thgpu, faceNormal0_thgpu, vertInfo0_thgpu, faceInfo0_thgpu):
    # Note the weight is per triangle, rather than per area.
    # If you want to uniformly sample from all the triangles, then larger triangles should
    # come with larger weight.
    assert len(faceVert0_thgpu.shape) == 3 and faceVert0_thgpu.shape[1:] == (3, 3)
    nFace = faceVert0_thgpu.shape[0]
    assert len(faceNormal0_thgpu.shape) == 2 and faceNormal0_thgpu.shape == (nFace, 3)
    assert len(faceCumsumWeight0_thgpu.shape) == 1
    assert faceCumsumWeight0_thgpu.shape[0] == nFace

    if faceCumsumWeight0_thgpu[-1] == 0:
        faceCumsumWeight0_thgpu = torch.from_numpy(
            np.arange(nFace).astype(np.float32) / nFace,
        ).to(faceCumsumWeight0_thgpu.device)

    # randomness
    # q_thgpu = torch.rand(num_sampling, dtype=torch.float32, device=faceVert0_thgpu.device) \
    #           * faceCumsumWeight0_thgpu[-1]
    # q_thgpu = (torch.div(
    #     torch.arange(num_sampling, dtype=torch.float64, device=faceVert0_thgpu.device) + 0.5,
    #     num_sampling * torch.ones(
    #         num_sampling, dtype=torch.float64, device=faceVert0_thgpu.device)
    # ) * faceCumsumWeight0_thgpu[-1].double()).float()
    q_thgpu = (
        torch.arange(num_sampling, dtype=torch.float32, device=faceVert0_thgpu.device)
        + 0.5
    ) / num_sampling * faceCumsumWeight0_thgpu[-1]
    pointFace0_thgpu = batched_binary_search_over_cumsumed_vec_thgpu(
        values=q_thgpu.detach().clone(), cumsumed_vec=faceCumsumWeight0_thgpu,
    ).long()
    pointFace0_thgpu[pointFace0_thgpu < 0] = 0
    pointFace0_thgpu[pointFace0_thgpu >= nFace] = nFace - 1
    # p1 = torch.rand(num_sampling, device=faceVert0_thgpu.device).float()
    # p2 = torch.rand(num_sampling, device=faceVert0_thgpu.device).float()
    # p1 = d2(np.arange(num_sampling, dtype=np.int32), 3)
    # p1 = determinedArbitraryPermutedVector2correctcorrect(np.arange(num_sampling, dtype=np.int32))
    p1 = (d2(np.arange(num_sampling, dtype=np.float32), 11) + 0.5) / float(num_sampling)
    p2 = d2(np.arange(num_sampling, dtype=np.float32), 13) / float(num_sampling)
    p1 = torch.from_numpy(p1).to(faceVert0_thgpu.device)
    p2 = torch.from_numpy(p2).to(faceVert0_thgpu.device)
    # Did not use the random seed to make things fixed due to its incorporation to
    # the training process.

    pLeft = torch.min(p1, p2)
    pRight = torch.max(p1, p2)
    pointBary0_thgpu = torch.stack([pLeft, pRight - pLeft, 1. - pRight], 1)

    # cache
    pointFaceVert0_thgpu = faceVert0_thgpu[pointFace0_thgpu, :, :]
    pointFaceNormal0_thgpu = faceNormal0_thgpu[pointFace0_thgpu, :]
    pointFaceNormal0_thgpu = pointFaceNormal0_thgpu / \
                             (torch.norm(pointFaceNormal0_thgpu, p=2, dim=1)[:, None] + 1.e-5)

    # point0_thgpu and pointNormal0_thgpu
    point0_thgpu = pointBary0_thgpu[:, 0, None] * pointFaceVert0_thgpu[:, 0, :] + \
                   pointBary0_thgpu[:, 1, None] * pointFaceVert0_thgpu[:, 1, :] + \
                   pointBary0_thgpu[:, 2, None] * pointFaceVert0_thgpu[:, 2, :]
    pointNormal0_thgpu = pointFaceNormal0_thgpu

    # pointVertInfo0_thgpu
    pointVertInfo0_thgpu = {}
    for k in vertInfo0_thgpu.keys():
        # assert len(vertInfo0_thgpu[k].shape) == 2 and vertInfo0_thgpu[k].shape[0] == nVert
        faceVertInfo0_thgpu = vertInfo2faceVertInfoTHGPU(vertInfo0_thgpu[k][None], face0_thgpu[None])[0]
        pointFaceVertInfo0_thgpu = faceVertInfo0_thgpu[pointFace0_thgpu]
        pointVertInfo0_thgpu[k] = pointBary0_thgpu[:, 0, None] * pointFaceVertInfo0_thgpu[:, 0, :] + \
                                  pointBary0_thgpu[:, 1, None] * pointFaceVertInfo0_thgpu[:, 1, :] + \
                                  pointBary0_thgpu[:, 2, None] * pointFaceVertInfo0_thgpu[:, 2, :]

    # pointFaceInfo0_thgpu
    pointFaceInfo0_thgpu = {}
    for k in faceInfo0_thgpu.keys():
        assert len(faceInfo0_thgpu[k].shape) == 2 and faceInfo0_thgpu[k].shape[0] == nFace
        pointFaceInfo0_thgpu[k] = faceInfo0_thgpu[k][pointFace0_thgpu, :]

    randInfo0_thgpu = {
        'q': q_thgpu,
        'p1': p1,
        'p2': p2,
        'pointFace': pointFace0_thgpu,
        'pointBary': pointBary0_thgpu,
    }

    return point0_thgpu, pointNormal0_thgpu, pointVertInfo0_thgpu, pointFaceInfo0_thgpu, \
        randInfo0_thgpu
