# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import general_algorithm_v1.cuda.general_algorithm as ga_cuda


def batched_binary_search_over_cumsumed_vec_thgpu(values, cumsumed_vec):
    # values: (V, )
    # cumsumed_vec: (C, )

    V = values.shape[0]
    returned_index = torch.zeros((V, ), dtype=torch.int32, device=values.device)
    return ga_cuda.batched_binary_search_over_cumsumed_vec(values, cumsumed_vec, returned_index)


def check_batched_binary_search_over_cumsumed_vec_thgpu():
    import torch
    import numpy as np
    a1 = np.linspace(-5, -4, 3)
    a2 = np.linspace(0, 1, 21)
    a3 = np.linspace(4, 5, 3)
    a = np.concatenate([a1, a2, a3], 0)
    b = np.linspace(0, 1, 11)
    cudaDevice = 'cuda:0'
    a = torch.from_numpy(a).to(cudaDevice)
    b = torch.from_numpy(b).to(cudaDevice)
    r = batched_binary_search_over_cumsumed_vec_thgpu(a, b)
    print(a)
    print(b)
    print(r)


if __name__ == '__main__':
    check_batched_binary_search_over_cumsumed_vec()
