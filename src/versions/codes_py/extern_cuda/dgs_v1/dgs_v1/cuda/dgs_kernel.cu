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

#define ERROR_CODE -999999999
#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))

namespace{
template <typename scalar_t>
__global__ void dgs_forward_cuda_kernel(
    const scalar_t* input,
    const scalar_t* grid,
    scalar_t* __restrict__ phi,
    scalar_t* __restrict__ phi_on_i,
    scalar_t* __restrict__ phi_on_j,
    scalar_t* __restrict__ debugging_info,
    int B,
    int C,
    int H,
    int W,
    int Q,
    int D
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= B * Q) {
        return;
    }
    const int b = g / Q;
    const int q = g - b * Q;

    const scalar_t i = grid[b * (Q * 2) + q * 2 + 1];
    const scalar_t j = grid[b * (Q * 2) + q * 2 + 0];

    const bool i_too_top = (i <= -1. + 1. / H);
    const bool i_too_bottom = (i >= 1. - 1. / H);
    const bool j_too_left = (j <= -1. + 1. / W);
    const bool j_too_right = (j >= 1. - 1. / W);

    for (int c = 0; c < C; c++) {
        const int bqc = b * (Q * C) + q * C + c;

        if (D >= 4) {
            const int base = 0;
            debugging_info[bqc * D + base + 0] = i_too_top;
            debugging_info[bqc * D + base + 1] = i_too_bottom;
            debugging_info[bqc * D + base + 2] = j_too_left;
            debugging_info[bqc * D + base + 3] = j_too_right;
        }
        if (D >= 6) {
            const int base = 4;
            debugging_info[bqc * D + base + 0] = j;
            debugging_info[bqc * D + base + 1] = i;
        }

        if ((i_too_top) && (j_too_left)) {
            phi[bqc] = input[b * (H * W * C) + 0 * (W * C) + 0 * C + c];
            phi_on_i[bqc] = 0;
            phi_on_j[bqc] = 0;
        } else if ((i_too_bottom) && (j_too_left)) {
            phi[bqc] = input[b * (H * W * C) + (H - 1) * (W * C) + 0 * C + c];
            phi_on_i[bqc] = 0;
            phi_on_j[bqc] = 0;
        } else if ((i_too_top) && (j_too_right)) {
            phi[bqc] = input[b * (H * W * C) + 0 * (W * C) + (W - 1) * C + c];
            phi_on_i[bqc] = 0;
            phi_on_j[bqc] = 0;
        } else if ((i_too_bottom) && (j_too_right)) {
            phi[bqc] = input[b * (H * W * C) + (H - 1) * (W * C) + (W - 1) * C + c];
            phi_on_i[bqc] = 0;
            phi_on_j[bqc] = 0;
        } else if ((!i_too_top) && (!i_too_bottom) && (!j_too_left) && (!j_too_right)) {
            const scalar_t ix = ((j + 1.f) * W - 1) / 2;
            const scalar_t iy = ((i + 1.f) * H - 1) / 2;
            const int top = MAX(MIN((int)iy, H - 2), 0);
            const int bottom = top + 1;
            const int left = MAX(MIN((int)ix, W - 2), 0);
            const int right = left + 1;
            const scalar_t phiA = input[b * (H * W * C) + top * (W * C) + left * C + c];
            const scalar_t phiB = input[b * (H * W * C) + top * (W * C) + right * C + c];
            const scalar_t phiC = input[b * (H * W * C) + bottom * (W * C) + left * C + c];
            const scalar_t phiD = input[b * (H * W * C) + bottom * (W * C) + right * C + c];
            const scalar_t nw = (right - ix) * (bottom - iy);
            const scalar_t ne = (ix - left) * (bottom - iy);
            const scalar_t sw = (right - ix) * (iy - top);
            const scalar_t se = (ix - left) * (iy - top);
            phi[bqc] =
                nw * phiA + ne * phiB +
                sw * phiC + se * phiD;
            phi_on_i[bqc] =
                (- (right - ix) * H / 2.) * phiA + (- (ix - left) * H / 2.) * phiB +
                (+ (right - ix) * H / 2.) * phiC + (+ (ix - left) * H / 2.) * phiD;
            phi_on_j[bqc] =
                (- (bottom - iy) * W / 2.) * phiA + (+ (bottom - iy) * W / 2.) * phiB +
                (- (iy - top) * W / 2.) * phiC + (+ (iy - top) * W / 2.) * phiD;
        } else if (i_too_top || i_too_bottom) {
            const scalar_t ix = ((j + 1.f) * W - 1) / 2;
            const int left = MAX(MIN((int)ix, W - 2), 0);
            const int right = left + 1;
            int row;
            if (i_too_top) {
                row = 0;
            } else {
                row = H - 1;
            }
            const scalar_t phiLeft = input[b * (H * W * C) + row * (W * C) + left * C + c];
            const scalar_t phiRight = input[b * (H * W * C) + row * (W * C) + right * C + c];
            const scalar_t w = (right - ix);
            const scalar_t e = (ix - left);
            phi[bqc] = w * phiLeft + e * phiRight;
            phi_on_i[bqc] = 0;
            phi_on_j[bqc] = (W / 2.) * phiRight - (W / 2.) * phiLeft;
        } else if (j_too_left || j_too_right) {
            const scalar_t iy = ((i + 1.f) * H - 1) / 2;
            const int top = MAX(MIN((int)iy, H - 2), 0);
            const int bottom = top + 1;
            int col;
            if (j_too_left) {
                col = 0;
            } else {
                col = W - 1;
            }
            const scalar_t phiTop = input[b * (H * W * C) + top * (W * C) + col * C + c];
            const scalar_t phiBottom = input[b * (H * W * C) + bottom * (W * C) + col * C + c];
            const scalar_t n = (bottom - iy);
            const scalar_t s = (iy - top);
            phi[bqc] = n * phiTop + s * phiBottom;
            phi_on_i[bqc] = (H / 2.) * phiBottom - (H / 2.) * phiTop;
            phi_on_j[bqc] = 0;
        }
    }

}

template <typename scalar_t>
__global__ void dgs_backward_cuda_kernel(
    const scalar_t* partialL_over_phi,
    const scalar_t* partialL_over_phiOverXyzCam,
    const scalar_t* grid,
    const scalar_t* fh_over_z,
    const scalar_t* fw_over_z,
    const scalar_t* yCamPerspQueryMap_over_z,
    const scalar_t* xCamPerspQueryMap_over_z,
    scalar_t* __restrict__ partialL_over_feat,
    scalar_t* __restrict__ debugging_info,
    int B,
    int C,
    int H,
    int W,
    int Q,
    int D
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= B * Q) {
        return;
    }
    const int b = g / Q;
    const int q = g - b * Q;

    const scalar_t i = grid[b * (Q * 2) + q * 2 + 1];
    const scalar_t j = grid[b * (Q * 2) + q * 2 + 0];

    const bool i_too_top = (i <= -1. + 1. / H);
    const bool i_too_bottom = (i >= 1. - 1. / H);
    const bool j_too_left = (j <= -1. + 1. / W);
    const bool j_too_right = (j >= 1. - 1. / W);

    for (int c = 0; c < C; c++) {
        const int bqc = b * (Q * C) + q * C + c;
        const scalar_t partialL_phiIJ = partialL_over_phi[bqc];
        const scalar_t partialL_phiIJOverXCam =
            partialL_over_phiOverXyzCam[bqc * 3 + 0];
        const scalar_t partialL_phiIJOverYCam =
            partialL_over_phiOverXyzCam[bqc * 3 + 1];
        const scalar_t partialL_phiIJOverZCam =
            partialL_over_phiOverXyzCam[bqc * 3 + 2];

        if (i_too_top && j_too_left) {
            const int h = 0;
            const int w = 0;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + h * (W * C) + w * C + c),
                partialL_phiIJ
            );
        } else if (i_too_bottom && j_too_left) {
            const int h = H - 1;
            const int w = 0;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + h * (W * C) + w * C + c),
                partialL_phiIJ
            );
        } else if (i_too_top && j_too_right) {
            const int h = 0;
            const int w = W - 1;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + h * (W * C) + w * C + c),
                partialL_phiIJ
            );
        } else if (i_too_bottom && j_too_right) {
            const int h = H - 1;
            const int w = W - 1;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + h * (W * C) + w * C + c),
                partialL_phiIJ
            );
        } else if ((!i_too_top) && (!i_too_bottom) && (!j_too_left) && (!j_too_right)) {
            const scalar_t ix = ((j + 1.f) * W - 1) / 2;
            const scalar_t iy = ((i + 1.f) * H - 1) / 2;
            const int top = MAX(MIN((int)iy, H - 2), 0);
            const int bottom = top + 1;
            const int left = MAX(MIN((int)ix, W - 2), 0);
            const int right = left + 1;
            const scalar_t alpha = iy - top;
            const scalar_t beta = ix - left;

            const scalar_t oneMinusBeta_over_ah = (1. - beta) * H / 2.;
            const scalar_t oneMinusAlpha_over_aw = (1. - alpha) * W / 2.;
            const scalar_t beta_over_ah = beta * H / 2.;
            const scalar_t alpha_over_aw = alpha * W / 2.;
            const scalar_t fh_over_z_here = fh_over_z[b * Q + q];
            const scalar_t fw_over_z_here = fw_over_z[b * Q + q];
            const scalar_t yCamPerspQueryMap_over_z_here = yCamPerspQueryMap_over_z[b * Q + q];
            const scalar_t xCamPerspQueryMap_over_z_here = xCamPerspQueryMap_over_z[b * Q + q];

            // partialL_over_phiA =
            //        partialL_phiIJ * partialPhiIJ_phiA ........................................... (1)
            //      + partialL_phiIJOverXCam * partialPhiIJOverXCam_phiA ........................... (2)
            //      + partialL_phiIJOverYCam * partialPhiIJOverYCam_phiA ........................... (3)
            //      + partialL_phiIJOverZCam * partialPhiIJOverZCam_phiA ........................... (4)

            // Accumulate from phi_ij to phi_A
            const scalar_t partialPhiIJ_phiA = (1 - alpha) * (1 - beta);
            const scalar_t partialPhiIJOverXCam_phiA = -oneMinusAlpha_over_aw * fw_over_z_here;
            const scalar_t partialPhiIJOverYCam_phiA = -oneMinusBeta_over_ah * fh_over_z_here;
            const scalar_t partialPhiIJOverZCam_phiA =
                (oneMinusBeta_over_ah * yCamPerspQueryMap_over_z_here) +
                (oneMinusAlpha_over_aw * xCamPerspQueryMap_over_z_here);
            const scalar_t partialL_phiA =
                partialL_phiIJ * partialPhiIJ_phiA +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiA +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiA +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiA;
            const int hA = top;
            const int wA = left;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hA * (W * C) + wA * C + c),
                partialL_phiA
            );

            // Accumulate from phi_ij to phi_B
            const scalar_t partialPhiIJ_phiB = (1 - alpha) * beta;
            const scalar_t partialPhiIJOverXCam_phiB = oneMinusAlpha_over_aw * fw_over_z_here;
            const scalar_t partialPhiIJOverYCam_phiB = -beta_over_ah * fh_over_z_here;
            const scalar_t partialPhiIJOverZCam_phiB =
                (beta_over_ah * yCamPerspQueryMap_over_z_here) +
                (-oneMinusAlpha_over_aw * xCamPerspQueryMap_over_z_here);
            const scalar_t partialL_phiB =
                partialL_phiIJ * partialPhiIJ_phiB +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiB +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiB +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiB;
            const int hB = top;
            const int wB = left + 1;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hB * (W * C) + wB * C + c),
                partialL_phiB
            );

            // Accumulate from phi_ij to phi_C
            const scalar_t partialPhiIJ_phiC = alpha * (1. - beta);
            const scalar_t partialPhiIJOverXCam_phiC = -alpha_over_aw * fw_over_z_here;
            const scalar_t partialPhiIJOverYCam_phiC = oneMinusBeta_over_ah * fh_over_z_here;
            const scalar_t partialPhiIJOverZCam_phiC =
                (-oneMinusBeta_over_ah * yCamPerspQueryMap_over_z_here) +
                (alpha_over_aw * xCamPerspQueryMap_over_z_here);
            const scalar_t partialL_phiC =
                partialL_phiIJ * partialPhiIJ_phiC +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiC +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiC +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiC;
            const int hC = top + 1;
            const int wC = left;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hC * (W * C) + wC * C + c),
                partialL_phiC
            );

            // Accumulate from phi_ij to phi_D
            const scalar_t partialPhiIJ_phiD = alpha * beta;
            const scalar_t partialPhiIJOverXCam_phiD = alpha_over_aw * fw_over_z_here;
            const scalar_t partialPhiIJOverYCam_phiD = beta_over_ah * fh_over_z_here;
            const scalar_t partialPhiIJOverZCam_phiD =
                (-beta_over_ah * yCamPerspQueryMap_over_z_here) +
                (-alpha_over_aw * xCamPerspQueryMap_over_z_here);
            const scalar_t partialL_phiD =
                partialL_phiIJ * partialPhiIJ_phiD +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiD +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiD +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiD;
            const int hD = top + 1;
            const int wD = left + 1;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hD * (W * C) + wD * C + c),
                partialL_phiD
            );
        } else if (i_too_top || i_too_bottom) {
            const scalar_t ix = ((j + 1.f) * W - 1) / 2;
            const int left = MAX(MIN((int)ix, W - 2), 0);
            const int right = left + 1;
            int row;
            if (i_too_top) {
                row = 0;
            } else {
                row = H - 1;
            }
            const scalar_t beta = ix - left;

            const scalar_t fw_over_z_here = fw_over_z[b * Q + q];
            const scalar_t xCamPerspQueryMap_over_z_here = xCamPerspQueryMap_over_z[b * Q + q];

            const scalar_t partialPhiIJ_phiLeft = (1 - beta);
            const scalar_t partialPhiIJOverXCam_phiLeft = -fw_over_z_here * W / 2.;
            const scalar_t partialPhiIJOverZCam_phiLeft = xCamPerspQueryMap_over_z_here * W / 2.;
            const scalar_t partialL_phiLeft =
                partialL_phiIJ * partialPhiIJ_phiLeft +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiLeft +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiLeft;
            const int hLeft = row;
            const int wLeft = left;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hLeft * (W * C) + wLeft * C + c),
                partialL_phiLeft
            );

            const scalar_t partialPhiIJ_phiRight = beta;
            const scalar_t partialPhiIJOverXCam_phiRight = fw_over_z_here * W / 2.;
            const scalar_t partialPhiIJOverZCam_phiRight = -xCamPerspQueryMap_over_z_here * W / 2.;
            const scalar_t partialL_phiRight =
                partialL_phiIJ * partialPhiIJ_phiRight +
                partialL_phiIJOverXCam * partialPhiIJOverXCam_phiRight +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiRight;
            const int hRight = row;
            const int wRight = right;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hRight * (W * C) + wRight * C + c),
                partialL_phiRight
            );

        } else if (j_too_left || j_too_right) {
            const scalar_t iy = ((i + 1.f) * H - 1) / 2;
            const int top = MAX(MIN((int)iy, H - 2), 0);
            const int bottom = top + 1;
            int col;
            if (j_too_left) {
                col = 0;
            } else {
                col = W - 1;
            }
            const scalar_t alpha = iy - top;

            const scalar_t fh_over_z_here = fh_over_z[b * Q + q];
            const scalar_t yCamPerspQueryMap_over_z_here = yCamPerspQueryMap_over_z[b * Q + q];

            const scalar_t partialPhiIJ_phiTop = (1 - alpha);
            const scalar_t partialPhiIJOverYCam_phiTop = -fh_over_z_here * H / 2.;
            const scalar_t partialPhiIJOverZCam_phiTop = yCamPerspQueryMap_over_z_here * H / 2.;
            const scalar_t partialL_phiTop =
                partialL_phiIJ * partialPhiIJ_phiTop +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiTop +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiTop;
            const int hTop = top;
            const int wTop = col;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hTop * (W * C) + wTop * C + c),
                partialL_phiTop
            );

            const scalar_t partialPhiIJ_phiBottom = alpha;
            const scalar_t partialPhiIJOverYCam_phiBottom = fh_over_z_here * H / 2.;
            const scalar_t partialPhiIJOverZCam_phiBottom = -yCamPerspQueryMap_over_z_here * H / 2.;
            const scalar_t partialL_phiBottom =
                partialL_phiIJ * partialPhiIJ_phiBottom +
                partialL_phiIJOverYCam * partialPhiIJOverYCam_phiBottom +
                partialL_phiIJOverZCam * partialPhiIJOverZCam_phiBottom;
            const int hBottom = bottom;
            const int wBottom = col;
            atomicAdd(
                partialL_over_feat + (b * (H * W * C) + hBottom * (W * C) + wBottom * C + c),
                partialL_phiBottom
            );
        }
    }
}

template <typename scalar_t>
__global__ void dgs3d_forward_cuda_kernel(
    const scalar_t* input,
    const scalar_t* grid,
    scalar_t* __restrict__ phi4,
    scalar_t* __restrict__ debugging_info,
    int B,
    int C,
    int Dp,
    int H,
    int W,
    int Q,
    int D
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= B * Q) {
        return;
    }
    const int b = g / Q;
    const int q = g - b * Q;

    const scalar_t k = grid[b * (Q * 3) + q * 3 + 2];  // z
    const scalar_t i = grid[b * (Q * 3) + q * 3 + 1];  // y
    const scalar_t j = grid[b * (Q * 3) + q * 3 + 0];  // x

    const bool k_too_back = (k <= -1. + 1. / Dp);
    const bool k_too_front = (k >= 1. - 1. / Dp);
    const bool i_too_top = (i <= -1. + 1. / H);
    const bool i_too_bottom = (i >= 1. - 1. / H);
    const bool j_too_left = (j <= -1. + 1. / W);
    const bool j_too_right = (j >= 1. - 1. / W);

    scalar_t gamma = 0, alpha = 0, beta = 0;
    int back = 0, front = 0, top = 0, bottom = 0, left = 0, right = 0;

    if ((!k_too_back) && (!k_too_front)) {
        const scalar_t t = ((k + 1.f) * Dp - 1) / 2;
        back = MAX(MIN((int)t, Dp - 2), 0);
        front = back + 1;
        gamma = t - back;
    } else if (k_too_back) {
        back = 0;
        front = 0;
        gamma = 0;
    } else if (k_too_front) {
        back = Dp - 1;
        front = Dp - 1;
        gamma = 0;
    }

    if ((!i_too_top) && (!i_too_bottom)) {
        const scalar_t t = ((i + 1.f) * H - 1) / 2;
        top = MAX(MIN((int)t, H - 2), 0);
        bottom = top + 1;
        alpha = t - top;
    } else if (i_too_top) {
        top = 0;
        bottom = 0;
        alpha = 0;
    } else if (i_too_bottom) {
        top = H - 1;
        bottom = H - 1;
        alpha = 0;
    }

    if ((!j_too_left) && (!j_too_right)) {
        const scalar_t t = ((j + 1.f) * W - 1) / 2;
        left = MAX(MIN((int)t, W - 2), 0);
        right = left + 1;
        beta = t - left;
    } else if (j_too_left) {
        left = 0;
        right = 0;
        beta = 0;
    } else if (j_too_right) {
        left = W - 1;
        right = W - 1;
        beta = 0;
    }

    for (int c = 0; c < C; c++) {
        const int bqc = b * (Q * C) + q * C + c;

        const scalar_t phiBackA = input[b * (Dp * H * W * C) + back * (H * W * C) +
                                        top * (W * C) + left * C + c];
        const scalar_t phiBackB = input[b * (Dp * H * W * C) + back * (H * W * C) +
                                        top * (W * C) + right * C + c];
        const scalar_t phiBackC = input[b * (Dp * H * W * C) + back * (H * W * C) +
                                        bottom * (W * C) + left * C + c];
        const scalar_t phiBackD = input[b * (Dp * H * W * C) + back * (H * W * C) +
                                        bottom * (W * C) + right * C + c];
        const scalar_t phiFrontA = input[b * (Dp * H * W * C) + front * (H * W * C) +
                                         top * (W * C) + left * C + c];
        const scalar_t phiFrontB = input[b * (Dp * H * W * C) + front * (H * W * C) +
                                         top * (W * C) + right * C + c];
        const scalar_t phiFrontC = input[b * (Dp * H * W * C) + front * (H * W * C) +
                                         bottom * (W * C) + left * C + c];
        const scalar_t phiFrontD = input[b * (Dp * H * W * C) + front * (H * W * C) +
                                         bottom * (W * C) + right * C + c];

        phi4[bqc * 4 + 0] =
            (1 - gamma) * (1 - alpha) * (1 - beta) * phiBackA +
            (1 - gamma) * (1 - alpha) * beta * phiBackB +
            (1 - gamma) * alpha * (1 - beta) * phiBackC +
            (1 - gamma) * alpha * beta * phiBackD +
            gamma * (1 - alpha) * (1 - beta) * phiFrontA +
            gamma * (1 - alpha) * beta * phiFrontB +
            gamma * alpha * (1 - beta) * phiFrontC +
            gamma * alpha * beta * phiFrontD;

        if ((!j_too_left) && (!j_too_right)) {
            const scalar_t phiLeftSum =
                (1 - gamma) * (1 - alpha) * phiBackA +
                (1 - gamma) * alpha * phiBackC +
                gamma * (1 - alpha) * phiFrontA +
                gamma * alpha * phiFrontC;
            const scalar_t phiRightSum =
                (1 - gamma) * (1 - alpha) * phiBackB +
                (1 - gamma) * alpha * phiBackD +
                gamma * (1 - alpha) * phiFrontB +
                gamma * alpha * phiFrontD;
            phi4[bqc * 4 + 1] = (phiRightSum - phiLeftSum) * W / 2.;
        }

        if ((!i_too_top) && (!i_too_bottom)) {
            const scalar_t phiTopSum =
                (1 - gamma) * (1 - beta) * phiBackA +
                (1 - gamma) * beta * phiBackB +
                gamma * (1 - beta) * phiFrontA +
                gamma * beta * phiFrontB;
            const scalar_t phiBottomSum =
                (1 - gamma) * (1 - beta) * phiBackC +
                (1 - gamma) * beta * phiBackD +
                gamma * (1 - beta) * phiFrontC +
                gamma * beta * phiFrontD;
            phi4[bqc * 4 + 2] = (phiBottomSum - phiTopSum) * H / 2.;
        }

        if ((!k_too_back) && (!k_too_front)) {
            const scalar_t phiBackSum =
                (1 - alpha) * (1 - beta) * phiBackA +
                (1 - alpha) * beta * phiBackB +
                alpha * (1 - beta) * phiBackC +
                alpha * beta * phiBackD;
            const scalar_t phiFrontSum =
                (1 - alpha) * (1 - beta) * phiFrontA +
                (1 - alpha) * beta * phiFrontB +
                alpha * (1 - beta) * phiFrontC +
                alpha * beta * phiFrontD;
            phi4[bqc * 4 + 3] = (phiFrontSum - phiBackSum) * Dp / 2.;
        }

    }
}

template <typename scalar_t>
__global__ void dgs3d_backward_cuda_kernel(
    const scalar_t* grad_phi4,
    const scalar_t* grid,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ debugging_info,
    int B,
    int Dp,
    int H,
    int W,
    int C,
    int Q,
    int D
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= B * Q) {
        return;
    }
    const int b = g / Q;
    const int q = g - b * Q;

    const scalar_t k = grid[b * (Q * 3) + q * 3 + 2];  // z
    const scalar_t i = grid[b * (Q * 3) + q * 3 + 1];  // y
    const scalar_t j = grid[b * (Q * 3) + q * 3 + 0];  // x

    const bool k_too_back = (k <= -1. + 1. / Dp);
    const bool k_too_front = (k >= 1. - 1. / Dp);
    const bool i_too_top = (i <= -1. + 1. / H);
    const bool i_too_bottom = (i >= 1. - 1. / H);
    const bool j_too_left = (j <= -1. + 1. / W);
    const bool j_too_right = (j >= 1. - 1. / W);

    scalar_t gamma = 0, alpha = 0, beta = 0;
    int back = 0, front = 0, top = 0, bottom = 0, left = 0, right = 0;

    if ((!k_too_back) && (!k_too_front)) {
        const scalar_t t = ((k + 1.f) * Dp - 1) / 2;
        back = MAX(MIN((int)t, Dp - 2), 0);
        front = back + 1;
        gamma = t - back;
    } else if (k_too_back) {
        back = 0;
        front = 0;
        gamma = 0;
    } else if (k_too_front) {
        back = Dp - 1;
        front = Dp - 1;
        gamma = 0;
    }

    if ((!i_too_top) && (!i_too_bottom)) {
        const scalar_t t = ((i + 1.f) * H - 1) / 2;
        top = MAX(MIN((int)t, H - 2), 0);
        bottom = top + 1;
        alpha = t - top;
    } else if (i_too_top) {
        top = 0;
        bottom = 0;
        alpha = 0;
    } else if (i_too_bottom) {
        top = H - 1;
        bottom = H - 1;
        alpha = 0;
    }

    if ((!j_too_left) && (!j_too_right)) {
        const scalar_t t = ((j + 1.f) * W - 1) / 2;
        left = MAX(MIN((int)t, W - 2), 0);
        right = left + 1;
        beta = t - left;
    } else if (j_too_left) {
        left = 0;
        right = 0;
        beta = 0;
    } else if (j_too_right) {
        left = W - 1;
        right = W - 1;
        beta = 0;
    }

    for (int c = 0; c < C; c++) {
        scalar_t toBackA = 0, toBackB = 0, toBackC = 0, toBackD = 0;
        scalar_t toFrontA = 0, toFrontB = 0, toFrontC = 0, toFrontD = 0;

        const int bqc = b * (Q * C) + q * C + c;
        const scalar_t grad_val = grad_phi4[bqc * 4 + 0];
        const scalar_t grad_x = grad_phi4[bqc * 4 + 1];
        const scalar_t grad_y = grad_phi4[bqc * 4 + 2];
        const scalar_t grad_z = grad_phi4[bqc * 4 + 3];

        toBackA += grad_val * ((1 - gamma) * (1 - alpha) * (1 - beta));
        toBackB += grad_val * ((1 - gamma) * (1 - alpha) * beta);
        toBackC += grad_val * ((1 - gamma) * alpha * (1 - beta));
        toBackD += grad_val * ((1 - gamma) * alpha * beta);
        toFrontA += grad_val * (gamma * (1 - alpha) * (1 - beta));
        toFrontB += grad_val * (gamma * (1 - alpha) * beta);
        toFrontC += grad_val * (gamma * alpha * (1 - beta));
        toFrontD += grad_val * (gamma * alpha * beta);

        if ((!j_too_left) && (!j_too_right)) {
            const scalar_t t = grad_x * (W / 2.);
            toBackA -= t * ((1 - gamma) * (1 - alpha));
            toBackC -= t * ((1 - gamma) * alpha);
            toFrontA -= t * (gamma * (1 - alpha));
            toFrontC -= t * (gamma * alpha);
            toBackB += t * ((1 - gamma) * (1 - alpha));
            toBackD += t * ((1 - gamma) * alpha);
            toFrontB += t * (gamma * (1 - alpha));
            toFrontD += t * (gamma * alpha);
        }

        if ((!i_too_top) && (!i_too_bottom)) {
            const scalar_t t = grad_y * (H / 2.);
            toBackA -= t * ((1 - gamma) * (1 - beta));
            toBackB -= t * ((1 - gamma) * beta);
            toFrontA -= t * (gamma * (1 - beta));
            toFrontB -= t * (gamma * beta);
            toBackC += t * ((1 - gamma) * (1 - beta));
            toBackD += t * ((1 - gamma) * beta);
            toFrontC += t * (gamma * (1 - beta));
            toFrontD += t * (gamma * beta);
        }

        if ((!k_too_back) && (!k_too_front)) {
            const scalar_t t = grad_z * (Dp / 2.);
            toBackA -= t * ((1 - alpha) * (1 - beta));
            toBackB -= t * ((1 - alpha) * beta);
            toBackC -= t * (alpha * (1 - beta));
            toBackD -= t * (alpha * beta);
            toFrontA += t * ((1 - alpha) * (1 - beta));
            toFrontB += t * ((1 - alpha) * beta);
            toFrontC += t * (alpha * (1 - beta));
            toFrontD += t * (alpha * beta);
        }

        atomicAdd(
            grad_input + b * (Dp * H * W * C) + back * (H * W * C)
                + top * (W * C) + left * C + c,
            toBackA
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + back * (H * W * C)
                + top * (W * C) + right * C + c,
            toBackB
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + back * (H * W * C)
                + bottom * (W * C) + left * C + c,
            toBackC
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + back * (H * W * C)
                + bottom * (W * C) + right * C + c,
            toBackD
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + front * (H * W * C)
                + top * (W * C) + left * C + c,
            toFrontA
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + front * (H * W * C)
                + top * (W * C) + right * C + c,
            toFrontB
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + front * (H * W * C)
                + bottom * (W * C) + left * C + c,
            toFrontC
        );
        atomicAdd(
            grad_input + b * (Dp * H * W * C) + front * (H * W * C)
                + bottom * (W * C) + right * C + c,
            toFrontD
        );
    }
}

}


std::vector<at::Tensor> dgs_forward_cuda(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi,
    at::Tensor phi_on_i,
    at::Tensor phi_on_j,
    at::Tensor debugging_info
) {
    assert(input.dim() == 4);
    assert(grid.dim() == 3);
    assert(phi.dim() == 3);
    assert(phi_on_i.dim() == 3);
    assert(phi_on_j.dim() == 3);
    assert(debugging_info.dim() == 4);

    const auto B = input.size(0);
    const auto H = input.size(1);
    const auto W = input.size(2);
    const auto C = input.size(3);
    const auto Q = grid.size(1);
    const auto D = debugging_info.size(3);

    assert(grid.size(0) == B);
    assert(grid.size(2) == 2);
    assert((phi.size(0) == B) && (phi.size(1) == Q) && (phi.size(2) == C));
    assert((phi_on_i.size(0) == B) && (phi_on_i.size(1) == Q) && (phi_on_i.size(2) == C));
    assert((phi_on_j.size(0) == B) && (phi_on_j.size(1) == Q) && (phi_on_j.size(2) == C));
    assert((debugging_info.size(0) == B) && (debugging_info.size(1) == Q) && (debugging_info.size(2) == C));

    // assert((float)(grid.max()) >= -1.);  Do not know how to do this. lazy
    // assert((float)(grid.min()) <= 1.);

    const int threads = 512;
    const dim3 blocks((B * Q - 1) / threads + 1);
    cudaSetDevice(input.get_device());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "dgs_forward_cuda", ([&] {
        dgs_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            grid.data<scalar_t>(),
            phi.data<scalar_t>(),
            phi_on_i.data<scalar_t>(),
            phi_on_j.data<scalar_t>(),
            debugging_info.data<scalar_t>(),
            B,
            C,
            H,
            W,
            Q,
            D
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in dgs_forward_cuda: %s\n", cudaGetErrorString(err));

    return {phi, phi_on_i, phi_on_j, debugging_info};
}

std::vector<at::Tensor> dgs_backward_cuda(
    at::Tensor partialL_over_phi,
    at::Tensor partialL_over_phiOverXyzCam,
    at::Tensor grid,
    at::Tensor fh_over_z,
    at::Tensor fw_over_z,
    at::Tensor yCamPerspQueryMap_over_z,
    at::Tensor xCamPerspQueryMap_over_z,
    at::Tensor partialL_over_feat,
    at::Tensor debugging_info
) {
    assert(partialL_over_phi.dim() == 3);
    assert(partialL_over_phiOverXyzCam.dim() == 4);
    assert(grid.dim() == 3);
    assert(fh_over_z.dim() == 2);
    assert(fw_over_z.dim() == 2);
    assert(yCamPerspQueryMap_over_z.dim() == 2);
    assert(xCamPerspQueryMap_over_z.dim() == 2);
    assert(debugging_info.dim() == 4);

    const auto B = partialL_over_phi.size(0);
    const auto H = partialL_over_feat.size(1);
    const auto W = partialL_over_feat.size(2);
    const auto C = partialL_over_phi.size(2);
    const auto Q = partialL_over_phi.size(1);
    const auto D = debugging_info.size(3);

    assert((partialL_over_phi.size(0) == B) && (partialL_over_phi.size(1) == Q)
            && (partialL_over_phi.size(2) == C));
    assert((partialL_over_phiOverXyzCam.size(0) == B) && (partialL_over_phiOverXyzCam.size(1) == Q) &&
            (partialL_over_phiOverXyzCam.size(2) == C) && (partialL_over_phiOverXyzCam.size(3) == 3));
    assert((grid.size(0) == B) && (grid.size(1) == Q) && (grid.size(2) == 2));
    assert((fh_over_z.size(0) == B) && (fh_over_z.size(1) == Q));
    assert((fw_over_z.size(0) == B) && (fw_over_z.size(1) == Q));
    assert((yCamPerspQueryMap_over_z.size(0) == B) && (yCamPerspQueryMap_over_z.size(1) == Q));
    assert((xCamPerspQueryMap_over_z.size(0) == B) && (xCamPerspQueryMap_over_z.size(1) == Q));
    assert((partialL_over_feat.size(0) == B) && (partialL_over_feat.size(1) == H) &&
           (partialL_over_feat.size(2) == W) && (partialL_over_feat.size(3) == C));
    assert((debugging_info.size(0) == B) && (debugging_info.size(1) == Q) && (debugging_info.size(2) == C));

    const int threads = 512;
    const dim3 blocks((B * Q - 1) / threads + 1);
    cudaSetDevice(partialL_over_phi.get_device());

    AT_DISPATCH_FLOATING_TYPES(partialL_over_phi.type(), "dgs_backward_cuda", ([&] {
        dgs_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            partialL_over_phi.data<scalar_t>(),
            partialL_over_phiOverXyzCam.data<scalar_t>(),
            grid.data<scalar_t>(),
            fh_over_z.data<scalar_t>(),
            fw_over_z.data<scalar_t>(),
            yCamPerspQueryMap_over_z.data<scalar_t>(),
            xCamPerspQueryMap_over_z.data<scalar_t>(),
            partialL_over_feat.data<scalar_t>(),
            debugging_info.data<scalar_t>(),
            B,
            C,
            H,
            W,
            Q,
            D
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in dgs_backward_cuda: %s\n", cudaGetErrorString(err));

    return {partialL_over_feat, debugging_info};
}

std::vector<at::Tensor> dgs3d_forward_cuda(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi4,
    at::Tensor debugging_info
) {
    assert(input.dim() == 5);
    assert(grid.dim() == 3);
    assert(phi4.dim() == 4);
    assert(debugging_info.dim() == 4);

    const auto B = input.size(0);
    const auto Dp = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto C = input.size(4);
    const auto Q = grid.size(1);
    const auto D = debugging_info.size(3);

    assert(grid.size(0) == B);
    assert(grid.size(2) == 3);
    assert((phi4.size(0) == B) && (phi4.size(1) == Q) && (phi4.size(2) == C) && (phi4.size(3) == 4));
    assert((debugging_info.size(0) == B) && (debugging_info.size(1) == Q) && (debugging_info.size(2) == C));

    const int threads = 512;
    const dim3 blocks((B * Q - 1) / threads + 1);
    cudaSetDevice(input.get_device());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "dgs3d_forward_cuda", ([&] {
        dgs3d_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            grid.data<scalar_t>(),
            phi4.data<scalar_t>(),
            debugging_info.data<scalar_t>(),
            B,
            C,
            Dp,
            H,
            W,
            Q,
            D
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in dgs3d_forward_cuda: %s\n", cudaGetErrorString(err));

    return {phi4, debugging_info};
}

std::vector<at::Tensor> dgs3d_backward_cuda(
    at::Tensor grad_phi4,
    at::Tensor pCam,
    at::Tensor grad_input,
    at::Tensor debugging_info
) {
    assert(grad_phi4.dim() == 4);
    assert(pCam.dim() == 3);
    assert(grad_input.dim() == 5);
    assert(debugging_info.dim() == 4);

    const auto B = grad_input.size(0);
    const auto Dp = grad_input.size(1);
    const auto H = grad_input.size(2);
    const auto W = grad_input.size(3);
    const auto C = grad_input.size(4);
    const auto Q = grad_phi4.size(1);
    const auto D = debugging_info.size(3);

    assert((grad_phi4.size(0) == B) && (grad_phi4.size(2) == C) && (grad_phi4.size(3) == 4));
    assert((pCam.size(0) == B) && (pCam.size(1) == Q) && (pCam.size(2) == 3));
    assert((debugging_info.size(0) == B) && (debugging_info.size(1) == Q) &&
           (debugging_info.size(2) == C));

    const int threads = 512;
    const dim3 blocks((B * Q - 1) / threads + 1);
    cudaSetDevice(grad_phi4.get_device());

    AT_DISPATCH_FLOATING_TYPES(grad_phi4.type(), "dgs3d_backward_cuda", ([&] {
        dgs3d_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_phi4.data<scalar_t>(),
            pCam.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            debugging_info.data<scalar_t>(),
            B,
            Dp,
            H,
            W,
            C,
            Q,
            D
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in dgs3d_backward_cuda: %s\n", cudaGetErrorString(err));

    return {grad_input, debugging_info};
}
