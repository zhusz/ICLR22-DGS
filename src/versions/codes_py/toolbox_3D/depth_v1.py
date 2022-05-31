# (tfconda)
import numpy as np
# from .mesh_v1 import vertInfo2faceVertInfoNP
from .mesh_surgery_v1 import trimVertGeneral


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


def depthMap2mesh(depthMap0, fScaleX, fScaleY, cutLinkThre):
    assert len(depthMap0.shape) == 2
    assert type(fScaleX) is float
    assert type(fScaleY) is float

    # vertCam0
    H = int(depthMap0.shape[0])
    W = int(depthMap0.shape[1])
    extremeH = 1. - 1. / float(H)
    extremeW = 1. - 1. / float(W)
    xi = np.linspace(-extremeW, extremeW, W).astype(np.float32)
    yi = np.linspace(-extremeH, extremeH, H).astype(np.float32)
    x, y = np.meshgrid(xi, yi)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = depthMap0.reshape(-1).copy()
    zMask = (z > 0) & np.isfinite(z)
    z[zMask == 0] = np.nan
    vertCamPersp0 = np.stack([x, y, z], 1)
    vertCam0 = np.stack([
        vertCamPersp0[:, 0] * vertCamPersp0[:, 2] / fScaleX,
        vertCamPersp0[:, 1] * vertCamPersp0[:, 2] / fScaleY,
        vertCamPersp0[:, 2],
    ], 1)

    # face0
    indMap0 = np.arange(H * W, dtype=np.int32).reshape((H, W))
    upLeft = np.stack([indMap0[:-1, :-1], indMap0[1:, :-1], indMap0[:-1, 1:]], 2)
    downRight = np.stack([indMap0[:-1, 1:], indMap0[1:, :-1], indMap0[1:, 1:]], 2)
    face0 = np.concatenate([upLeft.reshape((-1, 3)), downRight.reshape((-1, 3))], 0)

    # trim cutLinks due to nan
    # trim cutLinks due to distancing threshold
    nFace = int(face0.shape[0])
    faceFlag0 = np.ones((nFace, ), dtype=np.int32)
    faceVertCam0 = vertInfo2faceVertInfoNP(vertCam0[None], face0[None])[0]
    faceTriangleLen0 = np.stack([
        np.linalg.norm(faceVertCam0[:, 1, :] - faceVertCam0[:, 2, :], ord=2, axis=1),
        np.linalg.norm(faceVertCam0[:, 2, :] - faceVertCam0[:, 0, :], ord=2, axis=1),
        np.linalg.norm(faceVertCam0[:, 0, :] - faceVertCam0[:, 1, :], ord=2, axis=1),
    ], 1)
    faceTriangleLenMax0 = faceTriangleLen0.max(1)
    faceFlag0[np.isnan(faceTriangleLenMax0)] = -1
    if cutLinkThre is not None:
        assert type(cutLinkThre) is float
        faceFlag0[faceTriangleLenMax0 >= cutLinkThre] = -2
    vertCamOut0, faceOut0, tmp = trimVertGeneral(
        vertCam0, face0[faceFlag0 > 0], {'vertCamPersp0': vertCamPersp0})
    vertCamPerspOut0 = tmp['vertCamPersp0']

    out = {
        'vertCamPersp': vertCamPerspOut0,
        'vertCam0': vertCamOut0,
        'face0': faceOut0,
    }
    for v in out.values():
        assert np.all(np.isfinite(v))
    return out
