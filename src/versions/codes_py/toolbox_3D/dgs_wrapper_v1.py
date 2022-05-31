# (tfconda)
import torch
import torch.nn as nn
import dgs_v1.cuda.dgs as dgs_cuda
import numpy as np


class DGS2DLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pCamPersp, fScaleWidth, fScaleHeight):
        # input: b, c, h, w  # different from dgs_forward
        # pCamPersp: b, q, 3
        # fw, fh: (b, )
        phi, phi_on_i, phi_on_j, debugging_info = dgs_forward(
            input.permute(0, 2, 3, 1).contiguous(),
            pCamPersp[:, :, :2].contiguous(), dim_of_debugging_info=0)
        phi = phi.permute(0, 2, 1).contiguous()  # (B, C, Q)
        phi_on_i = phi_on_i.permute(0, 2, 1).contiguous()  # (B, C, Q)
        phi_on_j = phi_on_j.permute(0, 2, 1).contiguous()  # (B, C, Q)
        debugging_info = debugging_info.permute(0, 2, 1, 3).contiguous()  # (B, C, Q, D)
        fh_over_z = fScaleHeight[:, None] / pCamPersp[:, :, 2].detach()  # (B, Q)
        fw_over_z = fScaleWidth[:, None] / pCamPersp[:, :, 2].detach()  # (B, Q)
        # both two (B, Q) below
        yCamPerspQueryMap_over_z = pCamPersp[:, :, 1].detach() / pCamPersp[:, :, 2].detach()
        xCamPerspQueryMap_over_z = pCamPersp[:, :, 0].detach() / pCamPersp[:, :, 2].detach()
        # All three (B, C, Q) below
        phi_on_xCam = phi_on_j * fw_over_z[:, None, :]
        phi_on_yCam = phi_on_i * fh_over_z[:, None, :]
        phi_on_zCam = - phi_on_i * yCamPerspQueryMap_over_z[:, None, :] \
                      - phi_on_j * xCamPerspQueryMap_over_z[:, None, :]

        phi4 = torch.stack([phi, phi_on_xCam, phi_on_yCam, phi_on_zCam], 2)  # (B, C, 4, Q)
        ctx.save_for_backward(
            fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z,
            pCamPersp, torch.tensor(input.shape, dtype=torch.int32, device='cpu')
        )
        # debugging_info = torch.cat([  # (B, C, Q, D)
        #     debugging_info, phi_on_i[:, :, :, None], phi_on_j[:, :, :, None],
        # ], 3)
        return phi4

    @staticmethod
    def backward(ctx, grad_phi4):
        # grad_phi4: b, c, 4(1+3), q
        # fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z: b, q
        # pCamPersp: b, q, 3
        fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z, \
            pCamPersp, input_size = ctx.saved_tensors
        input_size = [int(s) for s in input_size]
        input_size = (input_size[0], input_size[2], input_size[3], input_size[1])
        grad_input, debugging_info = dgs_backward(
            input_size,
            grad_phi4[:, :, 0, :].permute(0, 2, 1).contiguous(),
            grad_phi4[:, :, 1:, :].permute(0, 3, 1, 2).contiguous(),
            pCamPersp[:, :, :2].contiguous(),
            fh_over_z, fw_over_z, yCamPerspQueryMap_over_z, xCamPerspQueryMap_over_z,
            0,
        )
        grad_input = grad_input.permute(0, 3, 1, 2).contiguous()
        return grad_input, None, None, None


dgs2dLayerApply = DGS2DLayerFunction.apply


class DGS2DLayer(nn.Module):
    def __init__(self):
        super(DGS2DLayer, self).__init__()

    def forward(self, input, grid, fScaleWidth, fScaleHeight):
        return dgs2dLayerApply(input, grid, fScaleWidth, fScaleHeight)


class DGSLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pCamPersp, fw, fh):
        # input: b, c, h, w  # different from dgs_forward
        # pCamPersp: b, q, 3
        # fw, fh: float scalar
        phi, phi_on_i, phi_on_j, debugging_info = dgs_forward(
            input.permute(0, 2, 3, 1).contiguous(),
            pCamPersp[:, :, :2].contiguous(), dim_of_debugging_info=0)
        phi = phi.permute(0, 2, 1).contiguous()  # (B, C, Q)
        phi_on_i = phi_on_i.permute(0, 2, 1).contiguous()  # (B, C, Q)
        phi_on_j = phi_on_j.permute(0, 2, 1).contiguous()  # (B, C, Q)
        debugging_info = debugging_info.permute(0, 2, 1, 3).contiguous()  # (B, C, Q, D)
        fh_over_z = fh / pCamPersp[:, :, 2].detach()  # (B, Q)
        fw_over_z = fw / pCamPersp[:, :, 2].detach()  # (B, Q)
        # both two (B, Q) below
        yCamPerspQueryMap_over_z = pCamPersp[:, :, 1].detach() / pCamPersp[:, :, 2].detach()
        xCamPerspQueryMap_over_z = pCamPersp[:, :, 0].detach() / pCamPersp[:, :, 2].detach()
        # All three (B, C, Q) below
        phi_on_xCam = phi_on_j * fw_over_z[:, None, :]
        phi_on_yCam = phi_on_i * fh_over_z[:, None, :]
        phi_on_zCam = - phi_on_i * yCamPerspQueryMap_over_z[:, None, :] \
                      - phi_on_j * xCamPerspQueryMap_over_z[:, None, :]

        phi4 = torch.stack([phi, phi_on_xCam, phi_on_yCam, phi_on_zCam], 2)  # (B, C, 4, Q)
        ctx.save_for_backward(
            fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z,
            pCamPersp, torch.tensor(input.shape, dtype=torch.int32, device='cpu')
        )
        # debugging_info = torch.cat([  # (B, C, Q, D)
        #     debugging_info, phi_on_i[:, :, :, None], phi_on_j[:, :, :, None],
        # ], 3)
        return phi4

    @staticmethod
    def backward(ctx, grad_phi4):
        # grad_phi4: b, c, 4(1+3), q
        # fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z: b, q
        # pCamPersp: b, q, 3
        fw_over_z, fh_over_z, xCamPerspQueryMap_over_z, yCamPerspQueryMap_over_z, \
            pCamPersp, input_size = ctx.saved_tensors
        input_size = [int(s) for s in input_size]
        input_size = (input_size[0], input_size[2], input_size[3], input_size[1])
        grad_input, debugging_info = dgs_backward(
            input_size,
            grad_phi4[:, :, 0, :].permute(0, 2, 1).contiguous(),
            grad_phi4[:, :, 1:, :].permute(0, 3, 1, 2).contiguous(),
            pCamPersp[:, :, :2].contiguous(),
            fh_over_z, fw_over_z, yCamPerspQueryMap_over_z, xCamPerspQueryMap_over_z,
            0,
        )
        grad_input = grad_input.permute(0, 3, 1, 2).contiguous()
        return grad_input, None, None, None


dgsLayerApply = DGSLayerFunction.apply


class DGSLayer(nn.Module):
    def __init__(self, fw, fh):
        super(DGSLayer, self).__init__()
        self.fw = fw
        self.fh = fh

    def forward(self, input, grid):
        return dgsLayerApply(input, grid, self.fw, self.fh)


def dgs_forward(input_thgpu, grid_thgpu, dim_of_debugging_info):
    # mode is always bilinear
    # padding is always border
    # align_corner is always False

    #Inputs
    # input_thgpu: (b, h, w, c)
    # grid_thgpu: (b, q, 2)  This is slightly different from the torch.nn.functional.grid_sample

    #Outputs
    # phi_[on_i|j_]thgpu: (b, q, c)
    # debugging_info: (b, q, c, d)

    assert len(input_thgpu.shape) == 4
    assert len(grid_thgpu.shape) == 3
    assert input_thgpu.is_cuda
    assert input_thgpu.shape[0] == grid_thgpu.shape[0] and input_thgpu.device == grid_thgpu.device
    assert grid_thgpu.shape[2] == 2

    b, h, w, c = input_thgpu.shape
    q = grid_thgpu.shape[1]
    d = dim_of_debugging_info

    assert input_thgpu.dtype == torch.float32
    assert grid_thgpu.dtype == torch.float32

    phi_thgpu = torch.zeros(b, q, c, dtype=torch.float32, device=input_thgpu.device)
    phi_on_i_thgpu = torch.zeros(b, q, c, dtype=torch.float32, device=input_thgpu.device)
    phi_on_j_thgpu = torch.zeros(b, q, c, dtype=torch.float32, device=input_thgpu.device)
    debugging_info_thgpu = torch.zeros(b, q, c, d, dtype=torch.float32, device=input_thgpu.device)
    phi_thgpu, phi_on_i_thgpu, phi_on_j_thgpu, debugging_info_thgpu = \
        dgs_cuda.dgs_forward(
            input_thgpu, grid_thgpu,
            phi_thgpu, phi_on_i_thgpu, phi_on_j_thgpu,
            debugging_info_thgpu,
        )

    return phi_thgpu, phi_on_i_thgpu, phi_on_j_thgpu, debugging_info_thgpu


def dgs_backward(
        input_size,
        partialL_over_phi_thgpu,
        partialL_over_phiOnXyzCam_thgpu,
        grid_thgpu,
        fh_over_z_thgpu,
        fw_over_z_thgpu,
        yCamPerspQueryMap_over_z_thgpu,
        xCamPerspQueryMap_over_z_thgpu,
        dim_of_debugging_info):
    # mode is always bilinear
    # padding is always zeros
    # align_corner is always False

    # input_size_thgpu: (4, ) representing b, h, w, c each (tuple)
    # partialL_over_phi_thgpu: (b, q, c)
    # partialL_over_phiOnXyzCam_thgpu: (b, q, c, 3(xyzCam))
    # grid_thgpu: (b, q, 2(i, j))
    # and the following four: (b, q)

    # partialL_over_feat_thgpu: (b, h, w, c)
    # debugging_info_thgpu: (b, q, c, d)

    assert type(input_size) is tuple
    assert len(input_size) == 4
    b, h, w, c = input_size
    q = partialL_over_phi_thgpu.shape[1]
    assert partialL_over_phi_thgpu.shape == (b, q, c)
    assert partialL_over_phiOnXyzCam_thgpu.shape == (b, q, c, 3)
    assert grid_thgpu.shape == (b, q, 2)
    for x in [fh_over_z_thgpu, fw_over_z_thgpu,
              yCamPerspQueryMap_over_z_thgpu, xCamPerspQueryMap_over_z_thgpu]:
        assert x.shape == (b, q)
    assert type(dim_of_debugging_info) is int
    d = dim_of_debugging_info

    for x in [partialL_over_phi_thgpu, partialL_over_phiOnXyzCam_thgpu, grid_thgpu,
              fh_over_z_thgpu, fw_over_z_thgpu,
              yCamPerspQueryMap_over_z_thgpu, xCamPerspQueryMap_over_z_thgpu]:
        assert x.dtype == torch.float32

    partialL_over_feat_thgpu = torch.zeros(b, h, w, c, dtype=torch.float32, device=grid_thgpu.device)
    debugging_info_thgpu = torch.zeros(b, q, c, d, dtype=torch.float32, device=grid_thgpu.device)
    partialL_over_feat_thgpu, debugging_info_thgpu = \
        dgs_cuda.dgs_backward(
            partialL_over_phi_thgpu, partialL_over_phiOnXyzCam_thgpu,
            grid_thgpu,
            fh_over_z_thgpu, fw_over_z_thgpu,
            yCamPerspQueryMap_over_z_thgpu, xCamPerspQueryMap_over_z_thgpu,
            partialL_over_feat_thgpu, debugging_info_thgpu
        )

    return partialL_over_feat_thgpu, debugging_info_thgpu


class DGS3DLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pCam, sizeX, sizeY, sizeZ):
        # input: b, c, d, h, w
        # pCam: b, q, 3
        # size[X|Y|Z]: scalars - default is 2 (Normalized Device Coordinate, -1 ~ 1)
        #   A different size[X|Y|Z] only affects the gradient (i.e. phi4[:, :, 1:, :])

        # phi4: b, c, 4(val, grad_x, grad_y, grad_z), q

        phi4, debugging_info = dgs3d_forward(
            input.permute(0, 2, 3, 4, 1).contiguous(),
            pCam.contiguous(),
            dim_of_debugging_info=0
        )
        assert sizeX > 0
        assert sizeY > 0
        assert sizeZ > 0
        phi4 = torch.stack([
            phi4[:, :, :, 0],
            phi4[:, :, :, 1] * (2. / sizeX),
            phi4[:, :, :, 2] * (2. / sizeY),
            phi4[:, :, :, 3] * (2. / sizeZ),
        ], 3)
        phi4 = phi4.permute(0, 2, 3, 1).contiguous()
        ctx.save_for_backward(
            pCam,
            torch.tensor(input.shape, dtype=torch.int32, device='cpu'),
            torch.tensor([sizeX, sizeY, sizeZ], dtype=torch.float32, device='cpu')
        )
        return phi4

    @staticmethod
    def backward(ctx, grad_phi4):
        # grad_phi4: b, c, 4(1+3), q
        # pCam: b, q, 3
        pCam, input_shape, sizeXYZ = ctx.saved_tensors
        input_shape = [int(s) for s in input_shape]
        input_shape = (input_shape[0], input_shape[2], input_shape[3], input_shape[4],
                       input_shape[1])  # (b, dp, h, w, c)
        sizeX, sizeY, sizeZ = float(sizeXYZ[0]), float(sizeXYZ[1]), float(sizeXYZ[2])
        grad_phi4 = grad_phi4.permute(0, 3, 1, 2).contiguous()
        grad_phi4 = torch.stack([
            grad_phi4[:, :, :, 0],
            grad_phi4[:, :, :, 1] * (sizeX / 2.),
            grad_phi4[:, :, :, 2] * (sizeY / 2.),
            grad_phi4[:, :, :, 3] * (sizeZ / 2.),
        ], 3)
        grad_input, debugging_info = dgs3d_backward(
            input_shape,
            grad_phi4,
            pCam,
            16
        )
        grad_input = grad_input.permute(0, 4, 1, 2, 3)
        return grad_input, None, None, None, None


dgs3dLayerApply = DGS3DLayerFunction.apply


class DGS3DLayer(nn.Module):
    def __init__(self, sizeX, sizeY, sizeZ):
        super(DGS3DLayer, self).__init__()
        self.sizeX, self.sizeY, self.sizeZ = sizeX, sizeY, sizeZ

    def forward(self, input, grid):
        return dgs3dLayerApply(input, grid, self.sizeX, self.sizeY, self.sizeZ)


def dgs3d_forward(input_thgpu, grid_thgpu, dim_of_debugging_info):
    # mode is always bilinear
    # padding is always border
    # align_corner is always False

    #Inputs
    # input_thgpu: (b, dp, h, w, c)
    # grid_thgpu: (b, q, 3)

    #Outputs
    # phi4: (b, q, c, 4)
    # debugging_info = (b, q, c, d)

    assert len(input_thgpu.shape) == 5
    assert len(grid_thgpu.shape) == 3
    assert input_thgpu.is_cuda
    assert input_thgpu.shape[0] == grid_thgpu.shape[0] and input_thgpu.device == grid_thgpu.device
    assert grid_thgpu.shape[2] == 3

    b, dp, h, w, c = input_thgpu.shape
    q = grid_thgpu.shape[1]
    d = dim_of_debugging_info

    assert input_thgpu.dtype == torch.float32
    assert grid_thgpu.dtype == torch.float32

    phi4_thgpu = torch.zeros(b, q, c, 4, dtype=torch.float32, device=input_thgpu.device)
    debugging_info_thgpu = torch.zeros(b, q, c, d, dtype=torch.float32, device=input_thgpu.device)
    phi4_thgpu, debugging_info = dgs_cuda.dgs3d_forward(
        input_thgpu, grid_thgpu, phi4_thgpu, debugging_info_thgpu
    )

    return phi4_thgpu, debugging_info_thgpu


def dgs3d_backward(input_shape, grad_phi4, pCam, dim_of_debugging_info):
    # mode is always bilinear
    # padding is always zeros
    # align_corner is always False

    # input_shape: (5, ) representing b, dp, h, w, c each (tuple)
    # grad_phi4: (b, q, c, 4)
    # pCam: (b, q, 3)
    # debugging_info_thgpu: (b, q, c, d)

    assert type(input_shape) is tuple
    assert len(input_shape) == 5
    b, dp, h, w, c = input_shape
    q = grad_phi4.shape[1]
    assert grad_phi4.shape == (b, q, c, 4)
    assert pCam.shape == (b, q, 3)
    assert type(dim_of_debugging_info) is int
    d = dim_of_debugging_info
    assert grad_phi4.dtype == torch.float32
    assert pCam.dtype == torch.float32

    grad_input = torch.zeros(b, dp, h, w, c, dtype=torch.float32, device=pCam.device)
    debugging_info = torch.zeros(b, q, c, d, dtype=torch.float32, device=pCam.device)
    grad_input, debugging_info = dgs_cuda.dgs3d_backward(
        grad_phi4, pCam,
        grad_input, debugging_info,
    )
    return grad_input, debugging_info
