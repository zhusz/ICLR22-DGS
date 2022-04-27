# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch  # Routines for numpy and torch


def ELU02cam0(elu0):
    assert len(elu0) == 9
    ex = elu0[0]
    ey = elu0[1]
    ez = elu0[2]
    lx = elu0[3]
    ly = elu0[4]
    lz = elu0[5]
    ux = elu0[6]
    uy = elu0[7]
    uz = elu0[8]
    u = np.array([ux, uy, uz], dtype=np.float32)

    z = np.array([lx - ex, ly - ey, lz - ez], dtype=np.float32)
    z = z / np.linalg.norm(z)
    x = np.cross(z, u)
    y = np.cross(z, x)
    assert np.linalg.norm(x) > 0.
    assert np.linalg.norm(y) > 0.
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    M = np.array([[x[0], x[1], x[2], -(x[0] * ex + x[1] * ey + x[2] * ez)],
                  [y[0], y[1], y[2], -(y[0] * ex + y[1] * ey + y[2] * ez)],
                  [z[0], z[1], z[2], -(z[0] * ex + z[1] * ey + z[2] * ez)],
                  [  0.,   0.,   0., 1.]], dtype=np.float32)

    assert M[-1, 0] == 0.
    assert M[-1, 1] == 0.
    assert M[-1, 2] == 0.
    assert M[-1, 3] == 1.
    return M  # which is cam0


def ELU2cam(elu):
    assert len(elu.shape) == 2 and elu.shape[1] == 9
    m = elu.shape[0]
    e = elu[:, :3]
    l = elu[:, 3:-3]
    u = elu[:, -3:]

    z = l - e
    z_norm = np.linalg.norm(z, ord=2, axis=1)
    # assert np.all(z_norm > 0)  # invalid cases will output nan
    z = z / z_norm[:, None]
    x = np.cross(z, u)
    y = np.cross(z, x)
    x_norm = np.linalg.norm(x, ord=2, axis=1)
    y_norm = np.linalg.norm(y, ord=2, axis=1)
    # assert np.all(x_norm > 0)
    # assert np.all(y_norm > 0)
    x = x / x_norm[:, None]
    y = y / y_norm[:, None]

    MList = [
        x[:, 0], x[:, 1], x[:, 2], -(x * e).sum(1),
        y[:, 0], y[:, 1], y[:, 2], -(y * e).sum(1),
        z[:, 0], z[:, 1], z[:, 2], -(z * e).sum(1),
        np.zeros((m, ), dtype=np.float32), np.zeros((m, ), dtype=np.float32),
        np.zeros((m, ), dtype=np.float32), np.ones((m, ), dtype=np.float32),
    ]
    M = np.stack(MList, 1).astype(np.float32).reshape((m, 4, 4))
    return M


modelViewLookAtRDF0 = ELU02cam0
# We always use cam0 to do: vertCam0 = np.matmul(vertWorld0, cam0[:3, :3].transpose()) + cam0[:3, -1]
# We always use cbm0 to d0: vertWorld0 = np.matmul(vertCam0, cbm0[:3, :3].transpose()) + cbm0[:3, -1]
# naming of cbm: camera back


def camSys2CamPerspSys0(pointCam0, focalLengthWidth, focalLengthHeight, winWidth, winHeight, zNear=1.e-6):
    assert len(pointCam0.shape) == 2 and pointCam0.shape[1] == 3
    assert type(focalLengthWidth) is float and type(focalLengthHeight) is float

    if type(pointCam0) is np.ndarray:
        point = pointCam0.copy()
        point[:, 0] = point[:, 0] / np.maximum(point[:, 2], zNear) * (float(focalLengthWidth) / winWidth * 2.)
        point[:, 1] = point[:, 1] / np.maximum(point[:, 2], zNear) * (float(focalLengthHeight) / winHeight * 2.)
        pointCamPersp0 = point
    elif type(pointCam0) is torch.Tensor:
        point = pointCam0.detach().clone()
        point[:, 0] = point[:, 0] / torch.clamp(point[:, 2], min=zNear) * (float(focalLengthWidth) / winWidth * 2.)
        point[:, 1] = point[:, 1] / torch.clamp(point[:, 2], min=zNear) * (float(focalLengthHeight) / winHeight * 2.)
        pointCamPersp0 = point
    else:
        raise TypeError('pointCam0 has a wrong type %s' % type(pointCam0))
    return pointCamPersp0


def camSys2CamPerspSysTHGPU(pointCam, focalLengthWidth, focalLengthHeight,
                            winWidth, winHeight, zNear=float(1.e-6)):
    return torch.stack([
        torch.div(pointCam[:, :, 0], torch.clamp(pointCam[:, :, 2], min=zNear)) *
                  (torch.div(focalLengthWidth, winWidth) * 2.)[:, None],
        torch.div(pointCam[:, :, 1], torch.clamp(pointCam[:, :, 2], min=zNear)) *
                  (torch.div(focalLengthHeight, winHeight) * 2.)[:, None],
        pointCam[:, :, 2],
    ], 2)


def camPerspSys2CamSys0(pointCamPersp0, focalLengthWidth, focalLengthHeight, winWidth, winHeight, zNear=1.e-6):
    assert len(pointCamPersp0.shape) == 2 and pointCamPersp0.shape[1] == 3
    assert type(focalLengthWidth) is float and type(focalLengthHeight) is float

    if type(pointCamPersp0) is np.ndarray:
        point = pointCamPersp0.copy()
        point[:, 0] = point[:, 0] / (float(focalLengthWidth) / winWidth * 2.) * np.maximum(point[:, 2], zNear)
        point[:, 1] = point[:, 1] / (float(focalLengthHeight) / winHeight * 2.) * np.maximum(point[:, 2], zNear)
        pointCam0 = point
    elif type(pointCamPersp0) is torch.Tensor:
        point = pointCamPersp0.detach().clone()
        point[:, 0] = point[:, 0] / (float(focalLengthWidth) / winWidth * 2.) * torch.clamp(point[:, 2], min=zNear)
        point[:, 1] = point[:, 1] / (float(focalLengthHeight) / winHeight * 2.) * torch.clamp(point[:, 2], min=zNear)
        pointCam0 = point
    else:
        raise TypeError('pointCam0 has a wrong type %s' % type(pointCamPersp0))
    return pointCam0


def worldSys2WorldNpone(pointWorld0, min, max):
    assert len(pointWorld0.shape) == 2 and pointWorld0.shape[1] == 3
    assert min.shape == (3, )
    assert max.shape == (3, )
    pointWorldNpone0 = (pointWorld0 - min[None, :]) / (max[None, :] - min[None, :]) * 2. - 1.
    return pointWorldNpone0


'''
Different dataset has different up / depth direction def, so avoid using this function.
Instead, implement according to this mode everytime when dealing with a different function.
def cbm02ELU0(cbmR0, cbmT0):
    assert abs(np.linalg.det(cbmR0) - 1.) < 1.e-4
    E0 = cbmT0.copy()
    tmp = np.dot(cbmR0, np.array([0., 0., 1.], dtype=np.float32) - 0.)  # minus 0 as the inverse translate
    lookDir0 = tmp / np.linalg.norm(tmp)
    L0 = E0 + 1. * lookDir0
    tmp = np.dot(cbmR0, np.array([0., -1., 0.], dtype=np.float32) - 0.)  # minus 0 as the inverse translate
    U0 = tmp / np.linalg.norm(tmp)
    U0Check = np.dot(U0, np.array([0., 0., 1.], dtype=np.float32)) > 0  # True if it is conventional
    return E0, L0, U0, U0Check
'''


# def getRotationMatrixBatchNP(rotationAxis, degrees):  # functional
#     return getRotationMatrixBatchTH(
#         torch.from_numpy(rotationAxis), torch.from_numpy(degrees)).detach().numpy()


def getRotationMatrixBatchTH(rotationAxis, degrees):   # functional
    # No Translation, single quaternion
    # Inputs:
    #   rotationsAxis: m * 3
    #   degrees: m  (Note this is degree representation, not radius representation!)
    # Outputs:
    #   Rs: m * 3 * 3
    # State Changes:
    #   None!

    assert len(rotationAxis.shape) == 2 and rotationAxis.shape[1] == 3
    assert len(degrees.shape) == 1 and rotationAxis.shape[0] == degrees.shape[0]

    norms = torch.norm(rotationAxis, p=2, dim=1)
    assert norms.min() > 0.
    normalized = torch.div(rotationAxis, norms[:, None])

    halfRadian = degrees * .5 * np.pi / 180.
    vCos = torch.cos(halfRadian)
    vSin = torch.sin(halfRadian)
    quat = torch.cat([vCos[:, None], vSin[:, None] * normalized], dim=1)
    rotMats = quat2matTH(quat)
    return rotMats


def getRotationMatrixBatchNP(rotationAxis, degrees, **kwargs):  # functional
    assert len(rotationAxis.shape) == 2 and rotationAxis.shape[1] == 3
    assert len(degrees.shape) == 1 and rotationAxis.shape[0] == degrees.shape[0]

    norms = np.linalg.norm(rotationAxis, ord=2, axis=1)
    assert norms.min() > 0.
    normalized = np.divide(rotationAxis, norms[:, None])

    halfRadian = degrees * .5 * np.pi / 180.
    vCos = np.cos(halfRadian)
    vSin = np.sin(halfRadian)
    quat = np.concatenate([vCos[:, None], vSin[:, None] * normalized], 1)
    rotMats = quat2matNP(quat)
    return rotMats


# def getRotationMatrixBatchNP(rotationsAxis, degrees, **kwargs):  # functional
#     cudaDevice = kwargs.get('cudaDevice', None)
#     rotationsAxis = torch.from_numpy(rotationsAxis)
#     degrees = torch.from_numpy(degrees)
#     if cudaDevice:
#         rotationsAxis = rotationsAxis.to(cudaDevice)
#         degrees = degrees.to(cudaDevice)
#     return getRotationMatrixBatchTH(
#         rotationsAxis, degrees,
#     ).detach().cpu().numpy()


def mat2quatNP(R):  # functional
    assert len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3
    w = ((1. + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) ** .5) * .5
    fwInverse = .25 / w  # 1/4w
    x = (R[:, 2, 1] - R[:, 1, 2]) * fwInverse
    y = (R[:, 0, 2] - R[:, 2, 0]) * fwInverse
    z = (R[:, 1, 0] - R[:, 0, 1]) * fwInverse
    q = np.stack([w, x, y, z], 1)
    return q


def quat2matTH(quat):  # functional
    assert len(quat.shape) == 2 and quat.shape[1] == 4

    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMats = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMats


def quat2matNP(quat):  # functional
    assert len(quat.shape) == 2 and quat.shape[1] == 4

    norm_quat = quat
    norm_quat = norm_quat / np.linalg.norm(norm_quat, ord=2, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.shape[0]

    w2, x2, y2, z2 = w ** 2, x ** 2, y ** 2, z ** 2
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMats = np.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], 1).reshape((B, 3, 3))
    return rotMats


if __name__ == '__main__':
    rep = 8
    aa = torch.tensor([0, 0, 1]).float()
    aa = torch.stack([aa for _ in range(rep)], 0)
    bb = torch.Tensor(range(rep)).float() * 360. / rep
    cc = getRotationMatrixBatchTHGPU(aa, bb)
    import ipdb
    ipdb.set_trace()
    print(1 + 1)
