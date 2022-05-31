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
