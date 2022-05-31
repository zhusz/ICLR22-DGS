import numpy as np


def obtain_bbox_tuple_via_seg0(seg0):  # functional. non-batched
    # input:
    # seg0 is HxW (ndim==2)
    # output:
    # (left, right, top, bottom) following the XDXD principle

    assert len(seg0.shape) == 2
    seg_h = seg0.sum(axis=0)
    left = np.argmax(seg_h > 0)  # findFirst
    # right = findLast(seg_h > 0)
    right = len(seg_h) - np.argmax((seg_h > 0)[::-1]) - 1  # findLast
    seg_v = seg0.sum(axis=1)
    top = np.argmax(seg_v > 0)  # findFirst
    # bottom = findLast(seg_v > 0)
    bottom = len(seg_v) - np.argmax((seg_v > 0)[::-1]) - 1  # findLast
    return (left, right, top, bottom)


def obtain_bbox_np_via_pose(pose0, vPose0=None, thre=0.01):  # functional. non-batched
    # input:
    # pose0 is n_pts * 2
    # vPose0 is n_pts
    # output:
    # bbox0: 4
    assert(pose0.ndim == 2 and pose0.shape[1] == 2)
    n_pts = pose0.shape[0]
    if vPose0 is not None:
        assert(vPose0.ndim == 1 and vPose0.shape[0] == n_pts)

    p = pose0.copy()
    if vPose0 is not None:
        vp = np.stack([vPose0, vPose0], axis=1)
        p[vp <= thre] = np.nan

    bbox0 = np.zeros((4,), np.float32)
    bbox0[0] = np.nanmin(p[:, 0])
    bbox0[1] = np.nanmax(p[:, 0])
    bbox0[2] = np.nanmin(p[:, 1])
    bbox0[3] = np.nanmax(p[:, 1])
    return bbox0


def obtain_bboxes_np_via_poses(pose, vPose=None, thre=0.01):  # functional
    # input:
    # pose is m * n_pts * 2 (m samples)
    # vPose is m * n_pts
    # output:
    # bbox: m * 4

    assert(pose.ndim == 3 and pose.shape[2] == 2)
    m = pose.shape[0]
    n_pts = pose.shape[1]
    if vPose is not None:
        assert(vPose.ndim == 2 and vPose.shape[0] == m and vPose.shape[1] == n_pts)

    p = pose.copy()  # copy the pose everytime, as we think this is not a runtime bottleneck.
    if vPose is not None:
        vp = np.stack([vPose, vPose], axis=2)
        p[vp <= thre] = np.nan

    bbox = np.zeros((m, 4), np.float32)
    bbox[:, 0] = np.nanmin(p[:, :, 0], axis=1)  # left
    bbox[:, 1] = np.nanmax(p[:, :, 0], axis=1)  # right
    bbox[:, 2] = np.nanmin(p[:, :, 1], axis=1)  # top
    bbox[:, 3] = np.nanmax(p[:, :, 1], axis=1)  # bottom
    return bbox
