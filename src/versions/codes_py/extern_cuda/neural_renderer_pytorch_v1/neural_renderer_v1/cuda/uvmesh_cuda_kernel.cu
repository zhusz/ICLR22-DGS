#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

/*
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
*/

#define GET_2NORM_SQUARE(x, y) ((x) * (x) + (y) * (y))
#define GET_BARY_0(xc, yc, x0, y0, x1, y1, x2, y2) (-((xc)-(x1)) * ((y2)-(y1)) + ((yc)-(y1)) * ((x2)-(x1))) / (-((x0)-(x1)) * ((y2)-(y1)) + ((y0)-(y1)) * ((x2)-(x1)))
#define GET_BARY_1(xc, yc, x0, y0, x1, y1, x2, y2) (-((xc)-(x2)) * ((y0)-(y2)) + ((yc)-(y2)) * ((x0)-(x2))) / (-((x1)-(x2)) * ((y0)-(y2)) + ((y1)-(y2)) * ((x0)-(x2)))
// GET_BARY_2 should be obtained by 1 - bary0 - bary1
#define MAX_FLOAT 999999999.9

namespace{
template <typename scalar_t>
__global__ void IUV2FBC_cuda_kernel(
        const int32_t* I,
        const scalar_t* U,
        const scalar_t* V,
        const scalar_t* All_U_norm,
        const scalar_t* All_V_norm,
        const int32_t* All_FaceIndices,
        const int32_t* All_Faces,
        int32_t* __restrict__ F,
        scalar_t* __restrict__ BC,
        uint8_t* __restrict__ flag_inside,
        int L,
        int batch_size,
        int image_height,
        int image_width,
        int n_aug_verts,
        int n_aug_faces) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }
    const int ii = I[i];
    const scalar_t uu = U[i];
    const scalar_t vv = V[i];
    if ((ii < 0) || (ii >= L) || (uu < 0) || (uu > 1) || (vv < 0) || (vv > 1)) {
        return;
    }
    for (int j = 0; j < n_aug_faces; j++) {
        if (ii == All_FaceIndices[j]) {
            const int k0 = All_Faces[j * 3 + 0];
            const int k1 = All_Faces[j * 3 + 1];
            const int k2 = All_Faces[j * 3 + 2];
            scalar_t bc0 = GET_BARY_0(uu, vv, All_U_norm[k0], All_V_norm[k0], All_U_norm[k1], All_V_norm[k1], All_U_norm[k2], All_V_norm[k2]);
            scalar_t bc1 = GET_BARY_1(uu, vv, All_U_norm[k0], All_V_norm[k0], All_U_norm[k1], All_V_norm[k1], All_U_norm[k2], All_V_norm[k2]);
            scalar_t bc2 = 1. - bc0 - bc1;
            if ((bc0 >= 0) && (bc1 >= 0) && (bc2 >= 0)) {
                // Finally find it!
                F[i] = j;
                BC[i * 3 + 0] = bc0;
                BC[i * 3 + 1] = bc1;
                BC[i * 3 + 2] = bc2;
                flag_inside[i] = 1; // 255 not processed. 0 outside. 1 inside (normal).
                return;
            }
        }
    }
    // If still did not get returned, we search for the nearest vertex
    scalar_t dist_square_min = MAX_FLOAT;
    int best_j;
    int best_k;
    for (int j = 0; j < n_aug_faces; j++) {
        if (ii == All_FaceIndices[j]) {
            const int k0 = All_Faces[j * 3 + 0];
            const int k1 = All_Faces[j * 3 + 1];
            const int k2 = All_Faces[j * 3 + 2];
            const scalar_t dist_square_0 = GET_2NORM_SQUARE(All_U_norm[k0] - uu, All_V_norm[k0] - vv);
            const scalar_t dist_square_1 = GET_2NORM_SQUARE(All_U_norm[k1] - uu, All_V_norm[k1] - vv);
            const scalar_t dist_square_2 = GET_2NORM_SQUARE(All_U_norm[k2] - uu, All_V_norm[k2] - vv);
            if (dist_square_0 < dist_square_min) {
                best_j = j;
                best_k = 0;
                dist_square_min = dist_square_0;
            }
            if (dist_square_1 < dist_square_min) {
                best_j = j;
                best_k = 1;
                dist_square_min = dist_square_1;
            }
            if (dist_square_2 < dist_square_min) {
                best_j = j;
                best_k = 2;
                dist_square_min = dist_square_2;
            }
        }
    }
    F[i] = best_j;
    BC[i * 3 + 0] = 0;
    BC[i * 3 + 1] = 0;
    BC[i * 3 + 2] = 0;
    BC[i * 3 + best_k] = 1;
    flag_inside[i] = 0; // 255 not processed. 0 outside. 1 inside (normal).
}
}

std::vector<at::Tensor> IUV2FBC_cuda(
        at::Tensor I,
        at::Tensor U,
        at::Tensor V,
        at::Tensor All_U_norm,
        at::Tensor All_V_norm,
        at::Tensor All_FaceIndices,
        at::Tensor All_Faces, // the augmented version of tris,
        at::Tensor F,
        at::Tensor BC,
        at::Tensor flag_inside,
        int L) {

    const auto batch_size = I.size(0);
    const auto image_height = I.size(1);
    const auto image_width = I.size(2);
    const auto n_aug_verts = All_U_norm.size(1);
    const auto n_aug_faces = All_FaceIndices.size(1);

    const int threads = 512;
    const dim3 blocks((batch_size * image_height * image_width - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(U.type(), "IUV2FBC_cuda", ([&] {
        IUV2FBC_cuda_kernel<scalar_t><<<blocks, threads>>>(
            I.data<int32_t>(),
            U.data<scalar_t>(),
            V.data<scalar_t>(),
            All_U_norm.data<scalar_t>(),
            All_V_norm.data<scalar_t>(),
            All_FaceIndices.data<int32_t>(),
            All_Faces.data<int32_t>(),
            F.data<int32_t>(),
            BC.data<scalar_t>(),
            flag_inside.data<uint8_t>(),
            L,
            batch_size,
            image_height,
            image_width,
            n_aug_verts,
            n_aug_faces
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_face_info_onto_map: %s\n", cudaGetErrorString(err));

    return {F, BC, flag_inside};
}
