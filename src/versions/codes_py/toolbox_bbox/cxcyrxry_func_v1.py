# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def croppingCxcyrxry0(a0, cxcyrxry0, padConst=None):  # We return the deep copy version to ensure it is functional
    assert len(a0.shape) >= 2
    assert len(cxcyrxry0) == 4
    cx = int(round(cxcyrxry0[0]))
    cy = int(round(cxcyrxry0[1]))
    rx = int(round(cxcyrxry0[2]))
    ry = int(round(cxcyrxry0[3]))
    H = a0.shape[0]
    W = a0.shape[1]

    if cy >= ry and cx >= rx and cy + ry <= H and cx + rx <= W:
        b0 = a0[cy - ry: cy + ry, cx - rx: cx + rx]
        return b0

    b0 = a0
    # requiring paddings cases:
    if cy - ry < 0:
        to_pad = np.stack([b0[0] for _ in range(ry - cy)], 0)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([to_pad, b0], 0)
        cy = ry

    if cx - rx < 0:
        to_pad = np.stack([b0[:, 0] for _ in range(rx - cx)], 1)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([to_pad, b0], 1)
        cx = rx

    if cy + ry > H:
        to_pad = np.stack([b0[-1] for _ in range(cy + ry - H)], 0)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([b0, to_pad], 0)

    if cx + rx > W:
        to_pad = np.stack([b0[:, -1] for _ in range(cx + rx - W)], 1)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([b0, to_pad], 1)

    b0 = b0[cy - ry : cy + ry, cx - rx: cx + rx].copy()  # Here we do deep copy
    return b0

