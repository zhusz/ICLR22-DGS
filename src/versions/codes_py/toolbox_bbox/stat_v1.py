import numpy as np
import math


def bbox_intersect(bboxA, bboxB):  # functional
    # both have the same shape: m * 4 (left, right, top, bottom)
    # return m * 4
    assert(bboxA.shape == bboxB.shape)
    assert(len(bboxA.shape) == 2)
    assert(bboxA.shape[1] == 4)
    bbox = np.zeros_like(bboxA)
    bbox[:, 0] = np.maximum(bboxA[:, 0], bboxB[:, 0])
    bbox[:, 1] = np.minimum(bboxA[:, 1], bboxB[:, 1])
    bbox[:, 2] = np.maximum(bboxA[:, 2], bboxB[:, 2])
    bbox[:, 3] = np.minimum(bboxA[:, 3], bboxB[:, 3])
    return bbox


def bbox_area(bbox):  # functional
    assert len(bbox.shape) == 2
    assert bbox.shape[1] == 4
    left = bbox[:, 0]
    right = np.maximum(left, bbox[:, 1])
    top = bbox[:, 2]
    bottom = np.maximum(top, bbox[:, 3])
    return (right - left) * (bottom - top)


def IOU(bboxA, bboxB):  # functional
    # both have the same shape: m * 4 (left, right, top, bottom)
    # return an np array with length m
    assert bboxA.shape == bboxB.shape
    assert len(bboxA.shape) == 2
    assert bboxA.shape[1] == 4
    Intersect = bbox_area(bbox_intersect(bboxA, bboxB))
    A = bbox_area(bboxA)
    B = bbox_area(bboxB)
    U = A + B - Intersect
    return Intersect / U
