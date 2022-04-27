/*
# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
*/

#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace{
template <typename scalar_t>
__global__ void batched_binary_search_over_cumsumed_vec_cuda_kernel(
    const scalar_t* values,
    const scalar_t* cumsumed_vec,
    int32_t* __restrict__ returned_index,
    int V,
    int C
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= V) {
        return;
    }

    // binary search
    int head = 0;
    int tail = C;
    scalar_t value = values[g];
    int mid = -2;
    while (head < tail - 1) {
        mid = (head + tail) / 2;
        if (value <= cumsumed_vec[mid]) {
            tail = mid;
        } else {
            head = mid;
        }
    }

    // local fine detailing
    int searched_index = -2;
    for (int i = head; i < tail; i++) {
        if (value <= cumsumed_vec[i]) {
            searched_index = i;
            break;
        }
    }
    if (searched_index < 0) {
        searched_index = tail;
    }

    // boundary case
    if ((head == 0) && (value < cumsumed_vec[head])) {
        returned_index[g] = -1;
    } else if ((tail >= C - 1) && (value > cumsumed_vec[tail])) {
        returned_index[g] = C;
    } else {
        returned_index[g] = searched_index;
    }
}
}

at::Tensor batched_binary_search_over_cumsumed_vec_cuda(
    at::Tensor values,
    at::Tensor cumsumed_vec,
    at::Tensor returned_index
) {
    assert(values.dim() == 1);
    assert(cumsumed_vec.dim() == 1);
    assert(returned_index.dim() == 1);

    const auto V = values.size(0);
    const auto C = cumsumed_vec.size(0);
    assert(V == returned_index.size(0));

    const int threads = 512;
    const dim3 blocks((V - 1) / threads + 1);
    cudaSetDevice(values.get_device());

    AT_DISPATCH_FLOATING_TYPES(values.type(), "batched_binary_search_over_cumsumed_vec_cuda", ([&] {
        batched_binary_search_over_cumsumed_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            values.data<scalar_t>(),
            cumsumed_vec.data<scalar_t>(),
            returned_index.data<int32_t>(),
            V,
            C
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in general_algorithm_cuda: %s\n", cudaGetErrorString(err));

    return returned_index;
}


