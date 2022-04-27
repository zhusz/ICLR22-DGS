# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from matplotlib.cm import get_cmap
# from ..toolbox_human_pose.obtain_ref_pairs_v1 import obtainRefPairs  # requires proper imports to codes_py in path
from ..toolbox_bbox.obtain_bbox_from_annotations_v1 import obtain_bbox_np_via_pose
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np


def to_heatmap(x, cmap=None):
    x = x.astype(np.float32)
    upperBound = x[np.isfinite(x)].max()
    lowerBound = x[np.isfinite(x)].min()
    x = (x - lowerBound) / (upperBound - lowerBound)
    cm = plt.get_cmap(cmap, 2 ** 16)
    return cm(x)[..., :3]


def getPltDraw(f):
    fig = plt.figure()
    ax = plt.gca()

    f(ax)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))  # It is ARGB!
    buf = buf[:, :, [1, 2, 3, 0]]
    assert buf[:, :, -1].min() == 255
    render = buf[:, :, :3].astype(np.float32) / 255.
    # plt.clf()
    plt.close()

    return render


def getImshow(x, vmin=None, vmax=None, cmap=None):
    f = lambda ax: ax.imshow(x, vmin=vmin, vmax=vmax, cmap=cmap)
    render = getPltDraw(f)
    return render


def getScatter(*args, **kwargs):
    f = lambda ax: ax.scatter(*args, **kwargs)
    render = getPltDraw(f)
    return render


def drawDepthDeltaSign(depthA, depthB, deltaThre):
    # depthA << depthB (abs > deltaThre) navy blue
    # depthA < depthB (abs < deltaThre) sky blue
    # depthA > depthB (abs < deltaThre) orange red
    # depthA >> depthB (abs > deltaThre) red
    # depthA == depthB yellow
    # other masked region: black
    color1 = np.array([0.07, 0.07, 0.4], dtype=np.float32)
    color2 = np.array([0.1, 0.3, 0.5], dtype=np.float32)
    color3 = np.array([0.5, 0.3, 0.1], dtype=np.float32)
    color4 = np.array([0.4, 0.07, 0.07], dtype=np.float32)
    color5 = np.array([0.5, 0.5, 0.25], dtype=np.float32)

    assert depthA.shape == depthB.shape
    assert len(depthA.shape) == 2
    assert deltaThre > 0

    delta = depthA - depthB
    cloth = np.zeros((delta.shape[0], delta.shape[1], 3), dtype=np.float32)

    mask1 = (delta <= -deltaThre)
    mask2 = ((delta > -deltaThre) & (delta < 0))
    mask3 = ((delta < deltaThre) & (delta > 0))
    mask4 = (delta >= deltaThre)
    mask5 = delta == 0
    for c in range(3):
        cloth[:, :, c][mask1] = color1[c]
        cloth[:, :, c][mask2] = color2[c]
        cloth[:, :, c][mask3] = color3[c]
        cloth[:, :, c][mask4] = color4[c]
        cloth[:, :, c][mask5] = color5[c]
    return cloth


def getFontHeightWidth(fontSize, font, len_txt):
    if font is cv2.FONT_HERSHEY_TRIPLEX:
        fontHeight = int(fontSize * 25.)
        fontWidth = int(fontHeight * 0.7 * len_txt)
    else:
        raise NotImplementedError('Unknown font %s' % font)
    return fontHeight, fontWidth


# All the functions in here are non-batched
def drawBoxXDXD(img0, bboxXDXD0, lineWidthFloat=None, rgb=np.array([1., 0., 0.]), txt=None):  # functional, non-batched
    assert len(bboxXDXD0.shape) == 1 and bboxXDXD0.shape[0] == 4
    if lineWidthFloat is None:
        lineWidthFloat = math.sqrt(img0.shape[0] ** 2 + img0.shape[1] ** 2) * 3. / 2000.
    lineWidth = int(math.ceil(lineWidthFloat))
    b = np.around(bboxXDXD0).astype(np.int32)
    imgNew0 = img0.copy()
    cv2.rectangle(imgNew0, (b[0], b[2]), (b[1], b[3]), rgb.tolist(), lineWidth)
    if txt is not None:
        fontMin = 0.5
        fontMax = 1.
        heightFloat = b[3] - b[2]
        fontSize = max(heightFloat / 100., fontMin)
        fontSize = min(fontSize, fontMax)
        fontHeight = int(fontSize * 25.)
        fontWidth = int(fontHeight * 0.7 * len(txt))
        cv2.rectangle(imgNew0, (b[0], b[2] - fontHeight), (b[0] + fontWidth, b[2]), rgb.tolist(), -1)
        cv2.putText(imgNew0, txt, (b[0], b[2]), cv2.FONT_HERSHEY_TRIPLEX, fontSize, [0, 0, 0])
    return imgNew0


def drawBoxCorner(img0, bboxCornerPix0, rgb=np.array([1., 0., 0.], dtype=np.float32), lineWidthFloat=None, txt=None):
    assert len(bboxCornerPix0.shape) == 2 and bboxCornerPix0.shape[0] == 8 and bboxCornerPix0.shape[1] in [2, 3]  # can either input the useless z or not input it
    # Note bboxCornerPix0 is already ready for drawing (xy npone, z depth which is not useful here)
    winHeight = img0.shape[0]
    winWidth = img0.shape[1]
    assert winHeight > 20
    assert winWidth > 20
    if lineWidthFloat is None:
        lineWidthFloat = math.sqrt(winHeight ** 2 + winWidth ** 2) * 3. / 2000. * 3

    imgNew0 = img0.copy()
    linePointPairsOrdering = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for linePointPair in linePointPairsOrdering:
        cv2.line(imgNew0,
                 (int(bboxCornerPix0[linePointPair[0], 0] + 0.5), int(bboxCornerPix0[linePointPair[0], 1] + 0.5)),
                 (int(bboxCornerPix0[linePointPair[1], 0] + 0.5), int(bboxCornerPix0[linePointPair[1], 1] + 0.5)),
                 color=rgb.tolist(),
                 thickness=int(lineWidthFloat + 0.5))

    if txt is not None:
        fontMin = 0.5
        fontMax = 1.
        heightFloat = bboxCornerPix0[:, 1].max() - bboxCornerPix0[:, 1].min()
        fontSize = max(heightFloat / 100., fontMin)
        fontSize = min(fontSize, fontMax)
        fontHeight = int(fontSize * 25.)
        fontWidth = int(fontHeight * 0.7 * len(txt))
        cv2.rectangle(imgNew0, (int(bboxCornerPix0[0, 0]), int(bboxCornerPix0[0, 1]) - fontHeight), (int(bboxCornerPix0[0, 0]) + fontWidth, int(bboxCornerPix0[0, 1])), rgb.tolist(), -1)
        cv2.putText(imgNew0, txt, (int(bboxCornerPix0[0, 0]), int(bboxCornerPix0[0, 1])), cv2.FONT_HERSHEY_TRIPLEX, fontSize, [0, 0, 0])
    return imgNew0


'''
def drawSkel2D(img, pose2D, theme, visible_label=None):  # functional, non-batched
    newImg = img.copy()
    assert len(pose2D.shape) == 2
    assert pose2D.shape[1] == 2
    n_pts = pose2D.shape[0]
    colorBaseCh = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]  # conventions: r g b c m y
    refPair, colorIdx = obtainRefPairs(theme, n_pts)
    bbox = obtain_bbox_np_via_pose(pose2D, visible_label)
    bbox_diag_len = math.sqrt((bbox[1] - bbox[0]) ** 2 + (bbox[3] - bbox[2]) ** 2)
    for i in range(refPair.shape[0]):
        if type(visible_label) is type(None) or (visible_label[refPair[i, 0]] and visible_label[refPair[i, 1]]):
            # drawRGBLine(img, pose2D[refPair[i, 0], 0], pose2D[refPair[i, 0], 1], pose2D[refPair[i, 1], 0], pose2D[refPair[i, 1], 1], colorBaseCh[colorIdx[refPair[i, 0]]])
            cv2.line(newImg,
                     (pose2D[refPair[i, 0], 0], pose2D[refPair[i, 0], 1]),
                     (pose2D[refPair[i, 1], 0], pose2D[refPair[i, 1], 1]),
                     [int(x * 255.) for x in colorBaseCh[colorIdx[refPair[i, 0]]]],
                     int(math.ceil(4 * bbox_diag_len / 1000.)))
    for i in range(n_pts):
        if type(visible_label) is type(None) or visible_label[i]:
            # drawPoint(img, pose2D[i, 0], pose2D[i, 1], colorBaseCh[colorIdx[i]])
            cv2.circle(newImg,
                       (pose2D[i, 0], pose2D[i, 1]),
                       int(math.ceil(10 * bbox_diag_len / 1000.)),
                       [int(x * 255.) for x in colorBaseCh[colorIdx[i]]],
                       -1)
    return newImg
'''


def drawPoint(img0, data0, vData0=None, vThre=0.01, radius=3, color=[1., 0., 0.]):  # functional, non-batched
    newImg0 = img0.copy()
    assert len(img0.shape) == 3 and img0.shape[2] == 3
    assert len(data0.shape) == 2 and data0.shape[1] == 2
    nPts = data0.shape[0]
    if vData0 is not None:
        assert len(vData0.shape) == 1 and vData0.shape[0] == nPts

    for j in range(nPts):
        if vData0 is None or vData0[j] > vThre:
            # assert data0[j, 0] != 0
            # assert data0[j, 1] != 0
            cv2.circle(newImg0,
                       (int(data0[j, 0]), int(data0[j, 1])),
                       int(radius + 0.5),
                       [int(x * 255.) for x in color],
                       -1)
    return newImg0


def drawSquareFilled(img0, data0, vData0=None, vThre=0.01, radius=3, color=[1., 0., 0.]):  # functional, non-batched
    newImg0 = img0.copy()
    assert len(img0.shape) == 3 and img0.shape[2] == 3
    assert len(data0.shape) == 2 and data0.shape[1] == 2
    nPts = data0.shape[0]
    if vData0 is not None:
        assert len(vData0.shape) == 1 and vData0.shape[0] == nPts

    for j in range(nPts):
        if vData0 is None or vData0[j] > vThre:
            # assert data0[j, 0] != 0
            # assert data0[j, 1] != 0
            cx = data0[j, 0]
            cy = data0[j, 1]
            pt1 = (cx - radius, cy - radius)
            pt2 = (cx - radius, cy + radius)
            pt3 = (cx + radius, cy + radius)
            pt4 = (cx + radius, cy - radius)
            pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
            cv2.drawContours(newImg0, [pts], 0, [int(x * 255.) for x in color], -1)
    return newImg0


def drawTriangleFilled(img0, data0, vData0=None, vThre=0.01, radius=3, color=[1., 0., 0.]):  # functional, non-batched
    newImg0 = img0.copy()
    assert len(img0.shape) == 3 and img0.shape[2] == 3
    assert len(data0.shape) == 2 and data0.shape[1] == 2
    nPts = data0.shape[0]
    if vData0 is not None:
        assert len(vData0.shape) == 1 and vData0.shape[0] == nPts

    for j in range(nPts):
        if vData0 is None or vData0[j] > vThre:
            # assert data0[j, 0] != 0
            # assert data0[j, 1] != 0
            cx = data0[j, 0]
            cy = data0[j, 1]
            pt1 = (cx, cy - radius)
            pt2 = (cx - radius / 1.732, cy + radius / 2.)
            pt3 = (cx + radius / 1.732, cy + radius / 2.)
            pts = np.array([pt1, pt2, pt3], dtype=np.int32)
            cv2.drawContours(newImg0, [pts], 0, [int(x * 255.) for x in color], -1)
    return newImg0


def main():
    winSize = 400
    x = 300
    y = 100
    a = np.ones((winSize, winSize, 3), dtype=np.float32)
    b = drawPoint(a, np.array([x, y], dtype=np.float32)[None], radius=10)
    import matplotlib.pyplot as plt
    plt.imshow(b)
    plt.show()


def drawSeg(img, seg, L=None, fontSize=None, fontColor=[0., 0., 0.], textList=None, ifOverlay=True):  # functional, non-batched
    # fontsize: None -> no texting / 0 -> automatic fontsize texting / others -> fixed fontsize texting
    assert img.ndim == 3
    assert seg.ndim == 2
    assert img.shape[-1] == 3
    assert img.shape[0] == seg.shape[0]
    assert img.shape[1] == seg.shape[1]
    assert 'float' in str(img.dtype)
    assert 'int' in str(seg.dtype)
    if textList is not None:
        assert textList[0] is None  # No descriptions for the background
    newImg = img.copy()
    if L is None:
        L = seg.max() + 1  # 0 is the background
    else:
        if textList is not None:
            L = len(textList)

    if ifOverlay:
        cmap = get_cmap('Spectral')
        for i in range(1, L):
            color_now = cmap(float(i - 1) / max(1, L - 1))
            bool_map = seg == i
            for c in range(3):
                channel_now = newImg[:, :, c]
                channel_now[bool_map] = 0.5 * channel_now[bool_map] + 0.5 * color_now[c]

    if fontSize == 0:
        img_diag_len = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
        fontSize = img_diag_len / 2000.

    if fontSize is not None:
        for i in range(1, L):
            bool_map = seg == i
            if np.any(bool_map):
                Y, X = np.where(bool_map)
                Ymean = int(round(Y.mean()))
                Xmean = int(round(X.mean()))
                if textList is None:
                    cv2.putText(newImg, '%d' % i, (Xmean, Ymean), cv2.FONT_HERSHEY_TRIPLEX, fontSize,
                                [255. * x for x in fontColor], lineType=cv2.LINE_AA)
                else:
                    cv2.putText(newImg, '%d-%s' % (i, textList[i]), (Xmean, Ymean), cv2.FONT_HERSHEY_TRIPLEX, fontSize,
                                [255. * x for x in fontColor], lineType=cv2.LINE_AA)

    newImg[newImg < 0.] = 0.
    newImg[newImg > 1.] = 1.
    return newImg


def drawMask(img, masks, rgbPool):
    # Inputs:
    #   img: (height, width, 3)
    #   masks: (nBox, height, width)
    #   rgbPool: (nBox, 3)

    assert img.ndim == 3
    assert masks.ndim == 3
    assert img.shape[-1] == 3
    assert img.shape[0] == masks.shape[1]
    assert img.shape[1] == masks.shape[2]
    assert 'float' in str(img.dtype)
    assert 'int' in str(masks.dtype)
    assert rgbPool.shape[0] == masks.shape[0] and rgbPool.shape[1] == 3 and len(rgbPool.shape) == 2
    nBox = rgbPool.shape[0]

    if 'bool' in str(masks.dtype):
        maskBools = masks > 0
    else:
        maskBools = masks
    newImg = img.copy()
    for k in range(nBox):
        newImg += (rgbPool[k][None, None, :] * maskBools[k][:, :, None])
    newImg[newImg > 2.] = 2.
    newImg /= 2.

    return newImg


def drawTrID(img, triangle_id, n_faces):  # functional, non-batched
    assert img.ndim == 3 and img.shape[2] == 3
    assert triangle_id.ndim == 2 and triangle_id.shape[0] == img.shape[0] and triangle_id.shape[1] == img.shape[1]
    newImg = img.copy()
    f_total = n_faces
    f_half = f_total / 2
    first_flag = np.logical_and(triangle_id >= 0, triangle_id < f_half)
    second_flag = triangle_id >= f_half
    R_channel = newImg[:, :, 0]
    G_channel = newImg[:, :, 1]
    R_channel[first_flag] = triangle_id[first_flag].astype(float) / float(f_half)
    G_channel[first_flag] = 0.
    R_channel[second_flag] = 1.
    G_channel[second_flag] = triangle_id[second_flag].astype(float) / float(f_half) - 1.0
    return newImg


if __name__ == '__main__':
    main()
