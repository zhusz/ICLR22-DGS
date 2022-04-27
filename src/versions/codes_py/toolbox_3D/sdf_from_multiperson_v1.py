# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sdf_from_multiperson_v1.cuda.sdf_csrc as sdf_csrc


def sdfTHGPU(queryPoint, vert, face, referencePoints, dimCudaIntBuffer):
    assert queryPoint.dtype is torch.float32
    assert vert.dtype is torch.float32
    assert face.dtype is torch.int32
    assert referencePoints.dtype is torch.float32

    sdf = torch.zeros_like(queryPoint[..., 0])
    projPoint = torch.zeros_like(queryPoint)
    cudaIntBuffer = torch.zeros(list(sdf.shape) + [dimCudaIntBuffer],
                                dtype=torch.int32).to(queryPoint.device)
    assert dimCudaIntBuffer >= referencePoints.shape[1]
    sdf, projPoint, cudaIntBuffer = sdf_csrc.sdf(
        queryPoint, vert, face, referencePoints, sdf, projPoint, cudaIntBuffer,
    )

    # import matplotlib.pyplot as plt
    # from codes_py.toolbox_show_draw.show3D_v1 import showPoint3D4
    # ax, plt = showPoint3D4(queryPoint[0, sdf[0] > 0][:1000], None, 'r', 'x')
    # ax, plt = showPoint3D4(queryPoint[0, sdf[0] < 0][:1000], ax, 'b', 'x')
    # plt.show()
    # import ipdb
    # ipdb.set_trace()

    return sdf, projPoint, cudaIntBuffer


def udfTHGPU(queryPoint, vert, face):
    assert queryPoint.dtype is torch.float32
    assert vert.dtype is torch.float32
    assert face.dtype is torch.int32

    udf = torch.zeros_like(queryPoint[..., 0])
    projPoint = torch.zeros_like(queryPoint)

    udf, projPoint = sdf_csrc.udf(
        queryPoint, vert, face, udf, projPoint,
    )

    return udf, projPoint


def signValueMatchingTHGPU(mismatchedSdf, radius):
    # if sdf0's sign and value is composed from different sources
    # (e.g. sign from CoReNet while value from multiperson UDF)
    # Then use this function to make the volume match
    # via looking at the neighbouring of radius, choosing the biggest gradient axis

    assert mismatchedSdf.dtype is torch.float32
    assert len(mismatchedSdf.shape) == 4

    matchedSdf = torch.zeros_like(mismatchedSdf)

    tmp = sdf_csrc.sign_value_matching(
        mismatchedSdf, int(radius), matchedSdf
    )
    matchedSdf = tmp[0]

    return matchedSdf
