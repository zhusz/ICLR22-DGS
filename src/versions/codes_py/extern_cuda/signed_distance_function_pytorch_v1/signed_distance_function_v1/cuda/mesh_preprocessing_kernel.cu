#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#define SMALLPI 3.14159265
#define EPSILONFLOAT 0.0000025
#define LARGEFLOAT 10000000000.
#define SMALLFLOAT 0.0000000001

#define SGN(x) ((x) >= 0? (1) : (0))  // Throughout the SDF, we always treat 0 as positive.
#define ABS(x) ((x) > 0 ? (x) : (-(x)))
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MAX3(x, y, z) (MAX((MAX((x), (y))), (z)))

#define DOT(xA, yA, zA, xB, yB, zB) (((xA) * (xB)) + ((yA) * (yB)) + ((zA) * (zB)))
#define NORM(x, y, z) (sqrt(DOT((x), (y), (z), (x), (y), (z))))
#define COS(xA, yA, zA, xB, yB, zB) ((DOT(xA, yA, zA, xB, yB, zB)) / (NORM(xA, yA, zA)) / (NORM(xB, yB, zB)))
#define CROSS_X(xA, yA, zA, xB, yB, zB) ((yA) * (zB) - (zA) * (yB))
#define CROSS_Y(xA, yA, zA, xB, yB, zB) ((zA) * (xB) - (xA) * (zB))
#define CROSS_Z(xA, yA, zA, xB, yB, zB) ((xA) * (yB) - (yA) * (xB))

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CPU(x) TORCH_CHECK(!(x.type().is_cuda()), #x "must not be a CUDA tensor")
#define CHECK_CPUINPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x);

namespace{
template <typename scalar_t>
__global__ void faceVert_find_next_cuda_kernel(const scalar_t* face,
                                           const scalar_t* validFaceFlag,
                                           scalar_t* __restrict__ nextFaceVertLoc,
                                           int m,
                                           int nFace) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nFace) return;
    const int b = i / nFace;
    const int f = i - b * nFace;

    const scalar_t* faceNow = face + b * nFace * 3;
    const scalar_t* validFaceFlagNow = validFaceFlag + b * nFace;
    for (int k = 0; k < 3; k++) {
        if (validFaceFlagNow[f] > 0) {
            int vertID = faceNow[f * 3 + k];
            bool alreadyFound = 0;
            for (int fOffset = 1; fOffset < nFace + 1; fOffset++) {
                if (alreadyFound) break;
                int fCurrent = (f + fOffset) % nFace;
                if (validFaceFlagNow[fCurrent] > 0) {
                    for (int kCurrent = 0; kCurrent < 3; kCurrent++) {
                        if (face[fCurrent * 3 + kCurrent] == vertID) {
                            nextFaceVertLoc[(i * 3 + k) * 2 + 0] = fCurrent;
                            nextFaceVertLoc[(i * 3 + k) * 2 + 1] = kCurrent;
                            alreadyFound = 1;
                            break;
                        }
                    }
                }
            }
        } else {
            nextFaceVertLoc[(i * 3 + k) * 2 + 0] = -1;
            nextFaceVertLoc[(i * 3 + k) * 2 + 1] = -1;
        }

    }
}

template<typename scalar_t>
__global__ void faceEdge_find_next_cuda_kernel(const scalar_t* face,
                                               const scalar_t* validFaceFlag,
                                               const scalar_t* nextFaceVertLoc,
                                               const scalar_t* sumFaceVert,
                                               scalar_t* __restrict__ nextFaceEdgeLoc,
                                               int m,
                                               int nFace) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nFace) return;
    const int b = i / nFace;
    const int f = i - b * nFace;

    const scalar_t* faceNow = face + b * nFace * 3;
    const scalar_t* validFaceFlagNow = validFaceFlag + b * nFace;
    const scalar_t* nextFaceVertLocNow = nextFaceVertLoc + b * nFace * 3 * 2;
    const scalar_t* sumFaceVertNow = sumFaceVert + b * nFace;
    for (int k = 0; k < 3; k++) {
        if (validFaceFlagNow[f] > 0) {
            // const int edgeNodeA = faceNow[(k + 1) % 3];
            const int edgeNodeB = faceNow[f * 3 + (k + 2) % 3];
            int currentFA = f;
            int currentKA = (k + 1) % 3;
            int nextFA = nextFaceVertLocNow[(currentFA * 3 + currentKA) * 2 + 0];
            int nextKA = nextFaceVertLocNow[(currentFA * 3 + currentKA) * 2 + 1];
            for (int s = 0; s < sumFaceVertNow[f * 3 + (k + 1) % 3]; s++) {
                currentFA = nextFA;
                currentKA = nextKA;
                nextFA = nextFaceVertLocNow[(currentFA * 3 + currentKA) * 2 + 0];
                nextKA = nextFaceVertLocNow[(currentFA * 3 + currentKA) * 2 + 1];
                const bool flagB0 = (faceNow[currentFA * 3 + 0] == edgeNodeB);
                const bool flagB1 = (faceNow[currentFA * 3 + 1] == edgeNodeB);
                const bool flagB2 = (faceNow[currentFA * 3 + 2] == edgeNodeB);
                if (validFaceFlagNow[currentFA] && (flagB0 + flagB1 + flagB2 == 1)) {
                    nextFaceEdgeLoc[(i * 3 + k) * 2 + 0] = currentFA;
                    if (currentKA != 0 && flagB0 == 0) nextFaceEdgeLoc[(i * 3 + k) * 2 + 1] = 0;
                    if (currentKA != 1 && flagB1 == 0) nextFaceEdgeLoc[(i * 3 + k) * 2 + 1] = 1;
                    if (currentKA != 2 && flagB2 == 0) nextFaceEdgeLoc[(i * 3 + k) * 2 + 1] = 2;
                    break;
                }
            }
        } else {
            nextFaceEdgeLoc[(i * 3 + k) * 2 + 0] = -1;
            nextFaceEdgeLoc[(i * 3 + k) * 2 + 1] = -1;
        }
    }
}

template<typename scalar_t>
__global__ void count_cycle_face_indexing_cuda_kernel(const scalar_t* nextFaceIndexingLoc,
                                                      const scalar_t* validFaceFlag,
                                                      scalar_t* __restrict__ sumFaceIndexingLoc,
                                                      int m,
                                                      int nFace) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nFace) return;
    const int b = i / nFace;
    const int f = i - b * nFace;
    if (!validFaceFlag[i]) {
        for (int k = 0; k < 3; k++) sumFaceIndexingLoc[i * 3 + k] = -1;  // No need to provide number
        return;
    }

    const scalar_t* nextFaceIndexingLocNow = nextFaceIndexingLoc + b * nFace * 3 * 2;
    const scalar_t* validFaceFlagNow = validFaceFlag + b * nFace;
    for (int k = 0; k < 3; k++) {
        int nextF = f;
        int nextK = k;
        int count = 0;
        int currentF, currentK;
        bool success = 1;
        while (count == 0 || nextF != f || nextK != k) {
            if (count == nFace) {
                sumFaceIndexingLoc[i * 3 + k] = -3;  // There should not be a dead loop here
                success = 0;
                break;
            }
            currentF = nextF;
            currentK = nextK;
            if (!validFaceFlagNow[currentF]) {
                sumFaceIndexingLoc[i * 3 + k] = -4;  // Should not get into the invalid part
                success = 0;
                break;
            }
            count++;
            nextF = nextFaceIndexingLocNow[(currentF * 3 + currentK) * 2 + 0];
            nextK = nextFaceIndexingLocNow[(currentF * 3 + currentK) * 2 + 1];
        }
        if (success) {
            sumFaceIndexingLoc[i * 3 + k] = count;
        }
    }
}

template<typename scalar_t>
__global__ void calc_face_edge_normal_cuda_kernel(const scalar_t* faceNormal,
                                                  const int32_t* validFaceFlag,
                                                  const int32_t* nextFaceEdgeLoc,
                                                  const int32_t* sumFaceEdge,
                                                  scalar_t* __restrict__ faceEdgeNormal,
                                                  int m,
                                                  int nFace) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nFace) return;
    const int b = i / nFace;
    const int f = i - b * nFace;

    const scalar_t* faceNormalNow = faceNormal + b * nFace * 3;
    const int32_t* validFaceFlagNow = validFaceFlag + b * nFace;
    const int32_t* nextFaceEdgeLocNow = nextFaceEdgeLoc + b * nFace * 3 * 2;
    const int32_t* sumFaceEdgeNow = sumFaceEdge + b * nFace * 3;
    for (int k = 0; k < 3; k++) {
        if (validFaceFlagNow[f] > 0) {
            int nextF = f;
            int nextK = k;
            int currentF, currentK;
            scalar_t nx = 0.;
            scalar_t ny = 0.;
            scalar_t nz = 0.;
            for (int s = 0; s < sumFaceEdgeNow[f * 3 + k]; s++) {
                currentF = nextF;
                currentK = nextK;
                nextF = nextFaceEdgeLocNow[(currentF * 3 + currentK) * 2 + 0];
                nextK = nextFaceEdgeLocNow[(currentF * 3 + currentK) * 2 + 1];
                if (validFaceFlagNow[currentF]) {
                    nx += faceNormalNow[currentF * 3 + 0];
                    ny += faceNormalNow[currentF * 3 + 1];
                    nz += faceNormalNow[currentF * 3 + 2];
                }
            }
            scalar_t normalNorm = NORM(nx, ny, nz);
            if (normalNorm >= SMALLFLOAT) {
                nx /= normalNorm;
                ny /= normalNorm;
                nz /= normalNorm;
                faceEdgeNormal[(i * 3 + k) * 3 + 0] = nx;
                faceEdgeNormal[(i * 3 + k) * 3 + 1] = ny;
                faceEdgeNormal[(i * 3 + k) * 3 + 2] = nz;
            } else {
                faceEdgeNormal[(i * 3 + k) * 3 + 0] = -LARGEFLOAT * 3;  // This should not happen
                faceEdgeNormal[(i * 3 + k) * 3 + 1] = -LARGEFLOAT * 3;
                faceEdgeNormal[(i * 3 + k) * 3 + 2] = -LARGEFLOAT * 3;
            }
        } else {
            faceEdgeNormal[(i * 3 + k) * 3 + 0] = -LARGEFLOAT;
            faceEdgeNormal[(i * 3 + k) * 3 + 1] = -LARGEFLOAT;
            faceEdgeNormal[(i * 3 + k) * 3 + 2] = -LARGEFLOAT;
        }
    }
}

}

at::Tensor faceVert_find_next_cuda(at::Tensor face,
                               at::Tensor validFaceFlag,
                               at::Tensor nextFaceVertLoc) {

    /*
        ALL INT!!!
        Input:
          face: m * nFace * 3(012)
          validFaceFlag: m * nFace

        Output:
          nextFaceVertLoc: m * nFace * 3(012) * 2(faceID/faceK)
    */

    CHECK_INPUT(face);
    TORCH_CHECK(face.dim() == 3 && face.size(2) == 3, "vertVert_find_next_find_next: face shape wrong");
    const auto m = face.size(0);
    const auto nFace = face.size(1);

    CHECK_INPUT(validFaceFlag);
    TORCH_CHECK(validFaceFlag.dim() == 2 && validFaceFlag.size(0) == m && validFaceFlag.size(1) == nFace,
             "verVert_find_next: validFaceFlag shape wrong");

    CHECK_INPUT(nextFaceVertLoc);
    TORCH_CHECK(nextFaceVertLoc.dim() == 4 && nextFaceVertLoc.size(0) == m && nextFaceVertLoc.size(1) == nFace &&
             nextFaceVertLoc.size(2) == 3 && nextFaceVertLoc.size(3) == 2, "vertVert_find_next: nextFaceVertLoc shape wrong");

    const int threads = 512;
    const dim3 blocks((m * nFace - 1) / threads + 1);

    AT_DISPATCH_INTEGRAL_TYPES(face.type(), "faceVert_find_next_cuda", ([&] {
        faceVert_find_next_cuda_kernel<scalar_t><<<blocks, threads>>>(
            face.data<scalar_t>(),
            validFaceFlag.data<scalar_t>(),
            nextFaceVertLoc.data<scalar_t>(),
            m,
            nFace
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in faceVert_find_next: %s\n", cudaGetErrorString(err));
    }
    return nextFaceVertLoc;
}

at::Tensor faceEdge_find_next_cuda(at::Tensor face,
                              at::Tensor validFaceFlag,
                              at::Tensor nextFaceVertLoc,
                              at::Tensor sumFaceVert,
                              at::Tensor nextFaceEdgeLoc) {

    /*
        Input:
          face: m * nFace * 3(012)
          validFaceFlag: m * nFace
          nextFaceVertLoc: m * nFace * 3(012) * 2(faceID/faceK)

        Output:
          nextFaceEdgeLoc: m * nFace * 3(122001) * 2(faceID/faceK)
    */
    CHECK_INPUT(face);
    TORCH_CHECK(face.dim() == 3 && face.size(2) == 3, "face shape wrong");
    const auto m = face.size(0);
    const auto nFace = face.size(1);

    CHECK_INPUT(validFaceFlag);
    TORCH_CHECK(validFaceFlag.dim() == 2 && validFaceFlag.size(0) == m && validFaceFlag.size(1) == nFace,
             "validFaceFlag shape wrong");

    CHECK_INPUT(nextFaceVertLoc);
    TORCH_CHECK(nextFaceVertLoc.dim() == 4 && nextFaceVertLoc.size(0) == m && nextFaceVertLoc.size(1) == nFace &&
             nextFaceVertLoc.size(2) == 3 && nextFaceVertLoc.size(3) == 2, "nextFaceVertLoc shape wrong");

    CHECK_INPUT(sumFaceVert);
    TORCH_CHECK(sumFaceVert.dim() == 3 && sumFaceVert.size(0) == m && sumFaceVert.size(1) == nFace &&
             sumFaceVert.size(2) == 3, "sumFaceVert shape wrong");

    CHECK_INPUT(nextFaceEdgeLoc);
    TORCH_CHECK(nextFaceEdgeLoc.dim() == 4 && nextFaceEdgeLoc.size(0) == m && nextFaceEdgeLoc.size(1) == nFace &&
             nextFaceEdgeLoc.size(2) == 3 && nextFaceEdgeLoc.size(3) == 2, "nextFaceEdgeLoc shape wrong");

    const int threads = 512;
    const dim3 blocks((m * nFace - 1) / threads + 1);
    AT_DISPATCH_INTEGRAL_TYPES(face.type(), "faceEdge_find_next_cuda", ([&] {
        faceEdge_find_next_cuda_kernel<scalar_t><<<blocks, threads>>>(
            face.data<scalar_t>(),
            validFaceFlag.data<scalar_t>(),
            nextFaceVertLoc.data<scalar_t>(),
            sumFaceVert.data<scalar_t>(),
            nextFaceEdgeLoc.data<scalar_t>(),
            m,
            nFace
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in faceEge_find_next: %s\n", cudaGetErrorString(err));
    }
    return nextFaceEdgeLoc;
}

at::Tensor count_cycle_face_indexing_cuda(at::Tensor nextFaceIndexingLoc,
                                          at::Tensor validFaceFlag,
                                          at::Tensor sumFaceIndexingLoc) {

    /*
        ALL INT!
        Input:
          nextFaceIndexingLoc: m * nFace * 3(012or122001) * 2(f/k)
          validFaceFlag: m * nFace

        Output:
          sumFaceIndexingLoc: m * nFace * 3(012or122001)
    */

    CHECK_INPUT(nextFaceIndexingLoc);
    const auto m = nextFaceIndexingLoc.size(0);
    const auto nFace = nextFaceIndexingLoc.size(1);
    TORCH_CHECK(nextFaceIndexingLoc.dim() == 4 && nextFaceIndexingLoc.size(2) == 3 && nextFaceIndexingLoc.size(3) == 2, "nextFaceIndexingLoc shape wrong");
    CHECK_INPUT(validFaceFlag);
    TORCH_CHECK(validFaceFlag.dim() == 2 && validFaceFlag.size(0) == m && validFaceFlag.size(1) == nFace, "validFaceFlag shape wrong");
    CHECK_INPUT(sumFaceIndexingLoc);
    TORCH_CHECK(sumFaceIndexingLoc.dim() == 3 && sumFaceIndexingLoc.size(0) == m && sumFaceIndexingLoc.size(1) == nFace && sumFaceIndexingLoc.size(2) == 3, "sumFaceIndexingLoc shape wrong");

    const int threads = 512;
    const dim3 blocks((m * nFace - 1) / threads + 1);

    AT_DISPATCH_INTEGRAL_TYPES(nextFaceIndexingLoc.type(), "count_cycle_face_indexing_cuda", ([&] {
        count_cycle_face_indexing_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nextFaceIndexingLoc.data<scalar_t>(),
            validFaceFlag.data<scalar_t>(),
            sumFaceIndexingLoc.data<scalar_t>(),
            m,
            nFace
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in count_cycle_face_indexing: %s\n", cudaGetErrorString(err));
    }
    return sumFaceIndexingLoc;
}

at::Tensor calc_face_edge_normal_cuda(at::Tensor faceNormal,
                                      at::Tensor validFaceFlag,
                                      at::Tensor nextFaceEdgeLoc,
                                      at::Tensor sumFaceEdge,
                                      at::Tensor faceEdgeNormal) {

    /*
        Input:
          faceNormal: m * nFace * 3(nxnynz)
          validFaceFlag: m * nFace
          nextFaceEdgeLoc: m * nFace * 3(122001) * 2(nextF/nextK)
          sumFaceEdge: m * nFace * 3(122001)

        Output:
          faceEdgeNormal: m * nFace * 3(122001) * 3(nxnynz)
    */
    const auto m = faceNormal.size(0);
    const auto nFace = faceNormal.size(1);

    CHECK_INPUT(faceNormal);
    TORCH_CHECK(faceNormal.dim() == 3 && faceNormal.size(0) == m && faceNormal.size(1) == nFace
             && faceNormal.size(2) == 3, "faceNormal shape wrong");

    CHECK_INPUT(validFaceFlag);
    TORCH_CHECK(validFaceFlag.dim() == 2 && validFaceFlag.size(0) == m && validFaceFlag.size(1) == nFace, "validFaceFlag shape wrong");

    CHECK_INPUT(nextFaceEdgeLoc);
    TORCH_CHECK(nextFaceEdgeLoc.dim() == 4 && nextFaceEdgeLoc.size(0) == m && nextFaceEdgeLoc.size(1) == nFace &&
             nextFaceEdgeLoc.size(2) == 3 && nextFaceEdgeLoc.size(3) == 2, "nextFaceEdgeLoc shape wrong");

    CHECK_INPUT(sumFaceEdge);
    TORCH_CHECK(sumFaceEdge.dim() == 3 && sumFaceEdge.size(0) == m && sumFaceEdge.size(1) == nFace &&
             sumFaceEdge.size(2) == 3, "sumFaceEdge shape wrong");

    CHECK_INPUT(faceEdgeNormal);
    TORCH_CHECK(faceEdgeNormal.dim() == 4 && faceEdgeNormal.size(0) == m && faceEdgeNormal.size(1) == nFace &&
             faceEdgeNormal.size(2) == 3 && faceEdgeNormal.size(3) == 3, "faceEdgeNormal shape wrong");

    const int threads = 512;
    const dim3 blocks((m * nFace - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faceNormal.type(), "calc_face_edge_normal_cuda", ([&] {
        calc_face_edge_normal_cuda_kernel<scalar_t><<<blocks, threads>>>(
            faceNormal.data<scalar_t>(),
            validFaceFlag.data<int32_t>(),
            nextFaceEdgeLoc.data<int32_t>(),
            sumFaceEdge.data<int32_t>(),
            faceEdgeNormal.data<scalar_t>(),
            m,
            nFace
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in calc_face_edge_normal: %s\n", cudaGetErrorString(err));
    }
    return faceEdgeNormal;
}


