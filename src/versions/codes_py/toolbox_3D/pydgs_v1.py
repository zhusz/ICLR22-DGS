import torch
import torch.nn.functional as F


def pydgs_forwardValOnly(input_thgpu, grid_thgpu):
    # Hessian autogradable grid_sample
    # mode is always bilinear
    # padding is always replicate
    # align_corner is always False

    # Inputs
    # input_thgpu: (b, c, h, w)
    # grid_thgpu: (b, q, 2)

    # Outputs
    # phi that is Hessian (second order) differentiable
    # size (b, q, c_dim) which is materially different from grid_saple (transpose(0, 2, 1) required)

    assert len(input_thgpu.shape) == 4
    assert len(grid_thgpu.shape) == 3
    assert input_thgpu.is_cuda
    assert input_thgpu.shape[0] == grid_thgpu.shape[0] and input_thgpu.device == grid_thgpu.device
    assert grid_thgpu.shape[2] == 2

    b, c, h, w = input_thgpu.shape
    assert h >= 5
    assert w >= 5  # typically your feature map won't be smaller than this.
    # This aims to avoid inputting b, h, w, c format
    q = grid_thgpu.shape[1]

    assert input_thgpu.dtype == torch.float32
    assert grid_thgpu.dtype == torch.float32

    tx = (grid_thgpu[:, :, 0] - (-1.)) / (1. - (-1.)) * w - 0.5
    ty = (grid_thgpu[:, :, 1] - (-1.)) / (1. - (-1.)) * h - 0.5
    xx = torch.clamp(torch.floor(tx), min=-1, max=w - 1).long()  # (b, q)
    yy = torch.clamp(torch.floor(ty), min=-1, max=h - 1).long()
    bb = torch.arange(b, dtype=torch.int64, device=yy.device)[:, None].repeat(1, q)
    beta = torch.clamp(tx - xx, min=0., max=1.)[:, :, None]
    alpha = torch.clamp(ty - yy, min=0., max=1.)[:, :, None]

    padded_input_thgpu = F.pad(input_thgpu, (1, 1, 1, 1), 'replicate')
    xx += 1
    yy += 1

    outA = padded_input_thgpu[bb, :, yy, xx]
    outB = padded_input_thgpu[bb, :, yy, xx + 1]
    outC = padded_input_thgpu[bb, :, yy + 1, xx]
    outD = padded_input_thgpu[bb, :, yy + 1, xx + 1]
    out = (1. - alpha) * ((1. - beta) * outA + beta * outB) + \
                 alpha * ((1. - beta) * outC + beta * outD)

    return out


def pydgs_forward3dValOnly(input_thgpu, grid_thgpu):
    # Hessian autogradable grid_sample 3D version
    # mode is always bilinear
    # padding is always replicate
    # align_corner is always False

    # Inputs
    # input_thgpu: (b, c, d, h, w)
    # grid_thgpu: (b, q, 3)

    # Outputs
    # phi that is Hessian (second order) differentiable
    # size (b, q, c_dim) which is materially different from grid_saple (transpose(0, 2, 1) required)

    assert len(input_thgpu.shape) == 5
    assert len(grid_thgpu.shape) == 3
    assert input_thgpu.is_cuda
    assert input_thgpu.shape[0] == grid_thgpu.shape[0] and input_thgpu.device == grid_thgpu.device
    assert grid_thgpu.shape[2] == 3

    b, c, d, h, w = input_thgpu.shape
    assert d >= 5
    assert h >= 5
    assert w >= 5  # typically your feature map won't be smaller than this.

    q = grid_thgpu.shape[1]

    assert input_thgpu.dtype == torch.float32
    assert grid_thgpu.dtype == torch.float32

    tx = (grid_thgpu[:, :, 0] - (-1.)) / (1. - (-1.)) * w - 0.5
    ty = (grid_thgpu[:, :, 1] - (-1.)) / (1. - (-1.)) * h - 0.5
    tz = (grid_thgpu[:, :, 2] - (-1.)) / (1. - (-1.)) * d - 0.5
    xx = torch.clamp(torch.floor(tx), min=-1, max=w - 1).long()  # (b, q)
    yy = torch.clamp(torch.floor(ty), min=-1, max=h - 1).long()
    zz = torch.clamp(torch.floor(tz), min=-1, max=d - 1).long()
    bb = torch.arange(b, dtype=torch.int64, device=yy.device)[:, None].repeat(1, q)
    beta = torch.clamp(tx - xx, min=0., max=1.)[:, :, None]
    alpha = torch.clamp(ty - yy, min=0., max=1.)[:, :, None]
    gamma = torch.clamp(tz - zz, min=0., max=1.)[:, :, None]
    padded_input_thgpu = F.pad(input_thgpu, (1, 1, 1, 1, 1, 1), 'replicate')
    xx += 1
    yy += 1
    zz += 1

    outA = padded_input_thgpu[bb, :, zz, yy, xx]
    outB = padded_input_thgpu[bb, :, zz, yy, xx + 1]
    outC = padded_input_thgpu[bb, :, zz, yy + 1, xx]
    outD = padded_input_thgpu[bb, :, zz, yy + 1, xx + 1]
    outFirstHalf = (1. - alpha) * ((1. - beta) * outA + beta * outB) + \
                          alpha * ((1. - beta) * outC + beta * outD)
    outE = padded_input_thgpu[bb, :, zz + 1, yy, xx]
    outF = padded_input_thgpu[bb, :, zz + 1, yy, xx + 1]
    outG = padded_input_thgpu[bb, :, zz + 1, yy + 1, xx]
    outH = padded_input_thgpu[bb, :, zz + 1, yy + 1, xx + 1]
    outSecondHalf = (1. - alpha) * ((1. - beta) * outE + beta * outF) + \
                           alpha * ((1. - beta) * outG + beta * outH)
    out = (1. - gamma) * outFirstHalf + gamma * outSecondHalf

    return out
