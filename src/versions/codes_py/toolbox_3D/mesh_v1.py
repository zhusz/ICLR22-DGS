# pytorch3d and compiled extern_cuda
import numpy as np
import torch
import neural_renderer_v1 as nr
import neural_renderer_v1.cuda.rasterize as rasterize_cuda
import neural_renderer_v1.cuda.derasterize as derasterize_cuda


def vertInfo2faceVertInfoNP(vertInfo, face):  # Please find the th version in neural_renderer
    # Inputs:
    #  vertInfo: batchSize * nVert * d (if you mean coordinates, then d is 2 or 3)
    #  face: batchSize * nFace * 3
    # Outputs:
    #  faceVertInfo: batchSize * nFace * 3 * d

    assert len(vertInfo.shape) == 3
    assert len(face.shape) == 3
    b = vertInfo.shape[0]
    nVert = vertInfo.shape[1]
    d = vertInfo.shape[2]
    assert face.shape[0] == b
    assert face.shape[2] == 3
    assert 'float' in str(vertInfo.dtype)
    assert 'int' in str(face.dtype)

    t = face + (np.arange(b, dtype=face.dtype) * nVert)[:, None, None]
    v = vertInfo.reshape((b * nVert, d))

    faceVertInfo = v[t]
    return faceVertInfo


def vertInfo2faceVertInfoTHGPU(vertInfo, face):
    # Inputs:
    #  vertInfo: batchSize * nVert * d (if you mean coordinates, then d is 2 or 3)
    #  face: batchSize * nFace * 3
    # Outputs:
    #  faceVertInfo: batchSize * nFace * 3 * d

    assert len(vertInfo.shape) == 3
    assert len(face.shape) == 3
    b = vertInfo.shape[0]
    nVert = vertInfo.shape[1]
    d = vertInfo.shape[2]
    assert face.shape[0] == b
    assert face.shape[2] == 3
    assert 'float' in str(vertInfo.dtype)
    assert 'int' in str(face.dtype)

    t = face + (torch.arange(b, dtype=face.dtype).to(device=face.device) * nVert)[:, None, None]
    v = vertInfo.reshape((b * nVert, d))

    faceVertInfo = v[t]
    return faceVertInfo


def zBufferFromFaceVertBatchTHGPU(faceVert, height, width=None, near=0.01, far=float('inf'), faceFlag=None):
    # The only difference compared to obtainTrisCoverIDBatchTHGPU is that faceVert has already been put into the required faceVert format

    # Check input shapes
    assert len(faceVert.shape) == 4
    assert faceVert.shape[2] == 3
    assert faceVert.shape[3] == 3

    assert type(height) is int

    if width is not None:
        assert width == height  # misleading, shit!
    L = height
    m = faceVert.shape[0]
    nTris = faceVert.shape[1]

    cudaDevice = faceVert.device

    if faceFlag is None:
        faceFlag = torch.from_numpy(np.ones((m, nTris), dtype=np.int32)).cuda(cudaDevice)
    else:
        assert len(faceFlag.shape) == 2 and faceFlag.shape[0] == m and faceFlag.shape[1] == faceVert.shape[1]
        assert faceFlag.device == cudaDevice
        assert 'int32' in str(faceFlag.dtype)

    # Run cuda
    faces = faceVert[:, :, [0, 2, 1], :]  # nr uses clock-wise face, which is different from conventional counter-clock-wise.
    triangleID = ((-1) * torch.ones((m, L, L), dtype=torch.int32)).cuda(cudaDevice)
    bary = torch.zeros((m, L, L, 3), dtype=torch.float32).cuda(cudaDevice)
    buffer = torch.ones((m, L, L), dtype=torch.float32).cuda(cudaDevice) * np.inf
    faceInvMap = torch.zeros((1,), dtype=torch.float32).cuda(cudaDevice)
    facesInv = torch.zeros_like(faces)
    triangleID, bary, buffer, faceInvMap = \
        rasterize_cuda.forward_face_index_map(faces, faceFlag, triangleID, bary,
                                              buffer, faceInvMap, facesInv,
                                              L, near, far, True, False, False)

    return buffer, triangleID, bary


def zBufferBatchTHGPU(verts, face, height, width=None, near=0.01, far=float('inf'), faceFlag=None):  # functional
    # Input sizes:
    # (m, n_verts, 3), (m, n_tris, 3), ...
    # optionally: trisFlag: (m, n_tris)
    # Output sizes:
    # (m, height, width) doubled, (m, height, width, 3).

    # Always the smaller, the closer

    # All the arrays are pyTorch GPU Tensors (we are not asserting on this)
    # No gradient can be used after calling this module (assume this is for visualization)

    # Check input shapes

    faceVert = nr.vertices_to_faces(verts, face)
    return zBufferFromFaceVertBatchTHGPU(faceVert, height, width, near, far, faceFlag)


def zBufferBatchNP(verts, face, cudaDevice, height, width=None, near=0.01, far=float('inf'), faceFlag=None):  # functional
    buffer_thgpu, triangleID_thgpu, bary_thgpu = zBufferBatchTHGPU(
        torch.from_numpy(verts).cuda(cudaDevice),
        torch.from_numpy(face).cuda(cudaDevice),
        height,
        width,
        near=near,
        far=far,
        faceFlag=torch.from_numpy(faceFlag).cuda(cudaDevice) if faceFlag is not None else None,
    )
    return buffer_thgpu.detach().cpu().numpy(),\
           triangleID_thgpu.detach().cpu().numpy(),\
           bary_thgpu.detach().cpu().numpy()


def placeVertInfoOntoClothTHGPU(triangle_id, bary, face, vertInfo, background):  # functional
    # Inputs and Outputs formats are exactly the same as the slow python version below.
    cudaDevice = triangle_id.device
    assert bary.device == cudaDevice
    assert face.device == cudaDevice
    assert vertInfo.device == cudaDevice
    if type(background) is float:
        background = background * torch.ones((vertInfo.shape[-1], )).cuda(cudaDevice)
    else:
        assert background.device == cudaDevice
    cloth = torch.zeros((triangle_id.shape[0], triangle_id.shape[1], triangle_id.shape[2], vertInfo.shape[-1])).cuda(cudaDevice)
    cloth = rasterize_cuda.forward_vert_info_onto_map(triangle_id, bary, face, vertInfo, background, cloth)
    return cloth


def placeVertInfoOntoClothNP(triangle_id, bary, face, vertInfo, background, cudaDevice):  # functional
    if type(background) is float:
        background = background * np.ones((vertInfo.shape[-1], ), dtype=np.float32)
    cloth = placeVertInfoOntoClothTHGPU(
        torch.from_numpy(triangle_id).cuda(cudaDevice),
        torch.from_numpy(bary).cuda(cudaDevice),
        torch.from_numpy(face).cuda(cudaDevice),
        torch.from_numpy(vertInfo).cuda(cudaDevice),
        torch.from_numpy(background).cuda(cudaDevice),
    ).detach().cpu().numpy()
    return cloth


def placeFaceInfoOntoClothTHGPU(triangle_id, faceInfo, background):  # functional
    cudaDevice = triangle_id.device
    assert faceInfo.device == cudaDevice
    if type(background) is float:
        background = background * torch.ones((faceInfo.shape[-1], ), device=cudaDevice)
    else:
        assert background.device == cudaDevice
    cloth = torch.zeros((triangle_id.shape[0], triangle_id.shape[1], triangle_id.shape[2], faceInfo.shape[-1]), device=cudaDevice)
    cloth = rasterize_cuda.forward_face_info_onto_map(triangle_id, faceInfo, background, cloth)
    return cloth


def placeFaceInfoOntoClothNP(triangle_id, faceInfo, background, cudaDevice):  # functional
    if type(background) is float:
        background = background * np.ones((faceInfo.shape[-1], ), dtype=np.float32)
    cloth = placeFaceInfoOntoClothTHGPU(
        torch.from_numpy(triangle_id).cuda(cudaDevice),
        torch.from_numpy(faceInfo).cuda(cudaDevice),
        torch.from_numpy(background).cuda(cudaDevice),
    ).detach().cpu().numpy()
    return cloth


def getEdgeMaskTHGPU(verts2D, face, triangleID, pixel_is_area, soft_mode, line_width_radius):
    facesXY = nr.vertices_to_faces(verts2D, face[:, :, [0, 2, 1]])  # nr use clock-wise as triangle
    edgeMask = torch.zeros_like(triangleID).float()
    edgeMask = rasterize_cuda.forward_get_edge_mask(facesXY,
                                                    triangleID,
                                                    edgeMask,
                                                    pixel_is_area,
                                                    soft_mode,
                                                    line_width_radius)
    return edgeMask


def getEdgeMaskNP(verts2D, face, triangleID, pixel_is_area, soft_mode,
                  line_width_radius, cudaDevice):  # functional
    edgeMask = getEdgeMaskTHGPU(torch.from_numpy(verts2D).cuda(cudaDevice),
                                torch.from_numpy(face).cuda(cudaDevice),
                                torch.from_numpy(triangleID).cuda(cudaDevice),
                                pixel_is_area,
                                soft_mode,
                                line_width_radius).detach().cpu().numpy()
    return edgeMask
