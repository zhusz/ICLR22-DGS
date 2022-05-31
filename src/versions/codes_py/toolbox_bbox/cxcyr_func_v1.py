import numpy as np
import cv2


def bboxes2cxcyrs(bboxes):  # functional
    # Input: bboxes: m * 4
    # Output: cxcyr: m * 3
    assert(bboxes.ndim == 2 and bboxes.shape[1] == 4)
    m = bboxes.shape[0]
    cxcyr = np.zeros((m, 3), dtype=np.float32)
    cxcyr[:, 0] = (bboxes[:, 0] + bboxes[:, 1]) * .5  # cx
    cxcyr[:, 1] = (bboxes[:, 2] + bboxes[:, 3]) * .5  # cy
    cxcyr[:, 2] = (np.sqrt((bboxes[:, 1] - bboxes[:, 0]) ** 2 + (bboxes[:, 3] - bboxes[:, 2]) ** 2)) / 2.  # radius. hence half
    return cxcyr


def cxcyrs2bboxes(cxcyrs):  # functional
    # Input: cxcyrs: m * 3
    # Output: bboxes: m * 4
    assert(cxcyrs.ndim == 2 and cxcyrs.shape[1] == 3)
    m = cxcyrs.shape[0]
    bboxes = np.zeros((m, 4), dtype=np.float32)
    bboxes[:, 0] = cxcyrs[:, 0] - cxcyrs[:, 2]
    bboxes[:, 1] = cxcyrs[:, 0] + cxcyrs[:, 2]
    bboxes[:, 2] = cxcyrs[:, 1] - cxcyrs[:, 2]
    bboxes[:, 3] = cxcyrs[:, 1] + cxcyrs[:, 2]
    return bboxes


def inflateCxcyrs(cxcyrs, radiusInflationRate):  # functional
    # Inputs:
    #   cxcyrs: m * 3
    #   radiusInflateionRate: float scalar
    # Outputs:
    #   outputCxcyrs: m * 3
    assert(cxcyrs.ndim == 2 and cxcyrs.shape[1] == 3)
    assert type(radiusInflationRate) is float
    outCxcyrs = cxcyrs.copy()
    outCxcyrs[:, 2] *= radiusInflationRate
    return outCxcyrs


def inflateBboxes(bboxes, radiusInflationRate):  # functional
    # Inputs:
    #   bboxes: m * 4
    #   radiusInflationRate: float scalar
    # Outputs:
    #   outBboxes: m * 4
    assert(bboxes.ndim == 2 and bboxes.shape[1] == 4)
    return cxcyrs2bboxes(inflateCxcyrs(bboxes2cxcyrs(bboxes), radiusInflationRate))


def normalizeLandmarksCxcyr(data, cxcyr):  # functional
    # Level 0 -> Level 2
    # We do not process vData here
    # Inputs:
    #   data: m * nPts * 2
    #   cxcyr:
    # Outputs:
    #   newData: m * nPts * 2
    assert len(data.shape) == 3 and data.shape[-1] == 2
    assert 'float' in str(data.dtype)
    assert np.all(np.logical_not(np.isnan(data)))  # data cannot contain nan (though not necessary)
    assert data.shape[0] == cxcyr.shape[0]
    assert len(cxcyr.shape) == 2
    assert cxcyr.shape[1] == 3
    assert 'float' in str(cxcyr.dtype)
    newData = np.zeros_like(data)
    newData[:, :, 0] = (data[:, :, 0] - cxcyr[:, None, 0]) / cxcyr[:, None, 2]
    newData[:, :, 1] = (data[:, :, 1] - cxcyr[:, None, 1]) / cxcyr[:, None, 2]
    return newData


def denormalizeLandmarksCxcyr(data, cxcyr):  # functional
    # Level 2 -> Level 0
    # We do not process vData here
    # Inputs:
    #   data: m * nPts * 2
    #   cxcyr:
    # Outputs:
    #   newData: m * nPts * 2
    assert len(data.shape) == 3 and data.shape[-1] == 2
    assert 'float' in str(data.dtype)
    assert np.all(np.logical_not(np.isnan(data)))  # data cannot contain nan (though not necessary)
    assert data.shape[0] == cxcyr.shape[0]
    assert len(cxcyr.shape) == 2
    assert cxcyr.shape[1] == 3
    assert 'float' in str(cxcyr.dtype)
    newData = np.zeros_like(data)
    newData[:, :, 0] = data[:, :, 0] * cxcyr[:, None, 2] + cxcyr[:, None, 0]
    newData[:, :, 1] = data[:, :, 1] * cxcyr[:, None, 2] + cxcyr[:, None, 1]
    return newData


def croppingCxcyr0(oath, a0, cxcyr0, padConst=None):  # non-functional to a0
    # Inputs:
    #   oath: str indicating that you know this is non-functional and non-batch.
    #   a0: H * W [*...]. To make it functional, pass a0.copy() into here
    #   cxcyr0: as long as we can retrieve cxcyr0[0/1/2], can be tuple / list / np.ndarray
    #   padConst: scalar of any type (its type will be ignored)
    # Outputs:
    #   b0: (2 * radius) * (2 * radius) [*...]
    assert oath == 'I understand that this function is non-functional and non-batched.'
    assert len(a0.shape) >= 2
    assert len(cxcyr0) == 3
    cx = int(round(cxcyr0[0]))
    cy = int(round(cxcyr0[1]))
    r = int(round(cxcyr0[2]))
    H = a0.shape[0]
    W = a0.shape[1]

    if cy >= r and cx >= r and cy + r <= H and cx + r <= W:
        b0 = a0[cy - r : cy + r, cx - r : cx + r]
        return b0

    b0 = a0
    # requiring paddings cases:
    if cy - r < 0:
        to_pad = np.stack([b0[0] for _ in range(r - cy)], 0)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([to_pad, b0], 0)
        cy = r

    if cx - r < 0:
        to_pad = np.stack([b0[:, 0] for _ in range(r - cx)], 1)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([to_pad, b0], 1)
        cx = r

    if cy + r > H:
        to_pad = np.stack([b0[-1] for _ in range(cy + r - H)], 0)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([b0, to_pad], 0)

    if cx + r > W:
        to_pad = np.stack([b0[:, -1] for _ in range(cx + r - W)], 1)
        if padConst is not None:
            to_pad = np.ones_like(to_pad) * padConst
        b0 = np.concatenate([b0, to_pad], 1)

    b0 = b0[cy - r : cy + r, cx - r : cx + r]
    return b0


def decroppingCxcyr0(oath, b0, cxcyr0, a0, interp):  # non-functional to a0
    # Inputs:
    #   oath: str indicating that you know this is non-functional and non-batch.
    #   b0: someSize * someSize [*...]. Must be square.
    #   cxcyr0: as ong as we can retrieve cxcyr0[0/1/2], can be tuple / list / np.ndarray
    #   padConst: scalar of any type (its type will be ignored)
    #   a0: H * W [*...]. To make it functional, pass a0.copy() into here.
    # Outputs:
    #   a0: H * W [*...]. Only non-functional function can output the same var from the input.
    assert oath == 'I understand that this function is non-functional and non-batched.'
    assert(b0.shape[0] == b0.shape[1])
    assert(b0.shape[2:] == a0.shape[2:])
    assert len(cxcyr0) == 3
    cx = int(round(cxcyr0[0]))
    cy = int(round(cxcyr0[1]))
    r = int(round(cxcyr0[2]))
    if b0.shape[0] != 2 * r or b0.shape[1] != 2 * r:
        b0 = cv2.resize(b0, (2 * r, 2 * r), interpolation=interp)

    left_in_a = cx - r
    right_in_a = cx + r
    top_in_a = cy - r
    bottom_in_a = cy + r

    if left_in_a >= 0 and right_in_a < a0.shape[1] and top_in_a >= 0 and bottom_in_a < a0.shape[0]:
        a0[top_in_a : bottom_in_a, left_in_a : right_in_a] = b0
    else:
        c0 = b0.copy()
        if not bottom_in_a < a0.shape[0]:
            c0 = c0[:2 * r - (bottom_in_a - a0.shape[0])]
            bottom_in_a = a0.shape[0]
        if not right_in_a < a0.shape[1]:
            c0 = c0[:, :2 * r - (right_in_a - a0.shape[1])]
            right_in_a = a0.shape[1]
        if not top_in_a >= 0:
            c0 = c0[-top_in_a:, :]
            top_in_a = 0
        if not left_in_a >= 0:
            c0 = c0[:, -left_in_a:]
            left_in_a = 0
        a0[top_in_a : bottom_in_a, left_in_a : right_in_a] = c0
    return a0
