# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from pytorch3d.ops.knn import knn_points


def packageCDF1(predPCxyz0, labelPCxyz0, prDistThre, **kwargs):
    cudaDevice = kwargs['cudaDevice']
    predPCxyz0_thgpu = torch.from_numpy(predPCxyz0).to(cudaDevice)
    labelPCxyz0_thgpu = torch.from_numpy(labelPCxyz0).to(cudaDevice)
    tmpP = knn_points(
        predPCxyz0_thgpu[None], labelPCxyz0_thgpu[None],
        K=1, return_nn=False, return_sorted=False)
    tmpR = knn_points(
        labelPCxyz0_thgpu[None], predPCxyz0_thgpu[None],
        K=1, return_nn=False, return_sorted=False)
    distP = (tmpP.dists[0, :, 0] ** 0.5).detach().cpu().numpy()
    distR = (tmpR.dists[0, :, 0] ** 0.5).detach().cpu().numpy()
    acc = distP.mean()
    compl = distR.mean()
    chamfer = 0.5 * (acc + compl)
    prec = (distP < prDistThre).astype(np.float32).mean()
    recall = (distR < prDistThre).astype(np.float32).mean()
    if prec == 0 and recall == 0:
        F1 = 0.
    else:
        F1 = 2. * prec * recall / (prec + recall)
    return {
        'acc': acc,
        'compl': compl,
        'chamfer': chamfer,
        'prec': prec,
        'recall': recall,
        'F1': F1,
        'distP': distP,  # not a scalar
        'distR': distR,  # not a scalar
    }


def packageDepth(depthPred0, depthLabel0):
    assert depthPred0.shape == depthLabel0.shape
    assert len(depthPred0.shape) == 2

    depth_pred = depthPred0
    depth_trgt = depthLabel0

    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics


def affinePolyfitWithNaN(pred, label):
    # fit pred onto label, and return the fitted pred
    # Caution: do not fit the unordered point cloud!

    assert pred.shape == label.shape
    maskPred = np.isfinite(pred)
    maskLabel = np.isfinite(label)
    mask = maskPred & maskLabel
    if mask.sum() == 0:
        a, b = np.nan, np.nan
    else:
        a, b = np.polyfit(pred[mask], label[mask], deg=1)
    return a, b, a * pred + b
