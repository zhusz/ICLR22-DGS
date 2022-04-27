/*
# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/JiangWenPL/multiperson

*/

#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>


#define SMALLPI 3.14159265
#define EPSILONFLOAT 0.0000025
#define LARGEFLOAT 999999999.

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
__device__ void checkProjToTriangle(scalar_t queryX, scalar_t queryY, scalar_t queryZ,
                                    scalar_t x0, scalar_t y0, scalar_t z0,
                                    scalar_t x1, scalar_t y1, scalar_t z1,
                                    scalar_t x2, scalar_t y2, scalar_t z2,
                                    scalar_t nx, scalar_t ny, scalar_t nz,
                                    bool* _ifSignPos, bool* _ifProjInsideTriangle, scalar_t* _distQuery2TrianglePlane,
                                    scalar_t* _projXNow, scalar_t* _projYNow, scalar_t* _projZNow) {

    const scalar_t queryEnDot = DOT(queryX - x0, queryY - y0, queryZ - z0,
                                    nx, ny, nz);  // This is the sdf candidate. its abs is the distance.

    *_ifSignPos = SGN(queryEnDot);
    *_distQuery2TrianglePlane = ABS(queryEnDot);
    // P = Q - (0N) (becuase: 0P = 0Q - 0N)
    *_projXNow = queryX - nx * queryEnDot;
    *_projYNow = queryY - ny * queryEnDot;
    *_projZNow = queryZ - nz * queryEnDot;

    const scalar_t angle01 = acos(COS(x0 - queryX, y0 - queryY, z0 - queryZ, x1 - queryX, y1 - queryY, z1 - queryZ));
    const scalar_t angle02 = acos(COS(x0 - queryX, y0 - queryY, z0 - queryZ, x2 - queryX, y2 - queryY, z2 - queryZ));
    const scalar_t angle12 = acos(COS(x1 - queryX, y1 - queryY, z1 - queryZ, x2 - queryX, y2 - queryY, z2 - queryZ));

    if (angle01 + angle02 + angle12 >= 2 * SMALLPI) {  // the projection is inside
        *_ifProjInsideTriangle = 1;
    } else {
        *_ifProjInsideTriangle = 0;
    }
}

template <typename scalar_t>
__device__ void checkProjToLine(scalar_t queryX, scalar_t queryY, scalar_t queryZ,
                                scalar_t xA, scalar_t yA, scalar_t zA,
                                scalar_t xB, scalar_t yB, scalar_t zB,
                                scalar_t nx, scalar_t ny, scalar_t nz,
                                bool* _ifSignPos, bool* _ifProjInsideLine, scalar_t* _distQuery2Line,
                                scalar_t* _projXNow, scalar_t* _projYNow, scalar_t* _projZNow) {

    *_ifSignPos = SGN(DOT(queryX - xA, queryY - yA, queryZ - zA, nx, ny, nz));

    scalar_t xAB = xB - xA;
    scalar_t yAB = yB - yA;
    scalar_t zAB = zB - zA;
    const scalar_t normAB = NORM(xAB, yAB, zAB);
    xAB /= normAB;
    yAB /= normAB;
    zAB /= normAB;

    const scalar_t normProj = DOT(queryX - xA, queryY - yA, queryZ - zA, xAB, yAB, zAB);

    if (normProj >= 0 && normProj <= normAB) {
        *_ifProjInsideLine = 1;
    } else {
        *_ifProjInsideLine = 0;
    }
    *_projXNow = xA + normProj * xAB;
    *_projYNow = yA + normProj * yAB;
    *_projZNow = zA + normProj * zAB;
    *_distQuery2Line = NORM(queryX - *_projXNow, queryY - *_projYNow, queryZ - *_projZNow);
}

template <typename scalar_t>
__global__ void sdf_tris_search_cuda_kernel(
        const scalar_t* queryPoints,
        const int32_t* trisIDOrderingBuffer,
        const int32_t* gridTrisStartingIndex,
        const scalar_t* faceVert,
        const scalar_t* faceNormal,
        const scalar_t* faceEdgeNormal,
        const scalar_t* faceVertNormal,
        const int32_t* validFaceFlag,
        scalar_t* __restrict__ sdfPos,
        scalar_t* __restrict__ sdfNeg,  // always positive in its value
        scalar_t* __restrict__ projPointPos,
        scalar_t* __restrict__ projPointNeg,
        scalar_t* __restrict__ projPointNormalPos,
        scalar_t* __restrict__ projPointNormalNeg,
        int32_t* __restrict__ projTrisIDPos,
        int32_t* __restrict__ projTrisIDNeg,
        scalar_t* __restrict__ debuggingInfoExport,
        int m,
        int nQuery,
        int nFace,
        int totalGrid,
        int lenGridX,
        int lenGridY,
        int lenGridZ,
        int dimDebug,
        scalar_t xmin,
        scalar_t xmax,
        scalar_t ymin,
        scalar_t ymax,
        scalar_t zmin,
        scalar_t zmax,
        scalar_t aGrid) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nQuery) {
        return;
    }
    const int b = i / nQuery;

    const scalar_t queryX = queryPoints[i * 3 + 0];
    const scalar_t queryY = queryPoints[i * 3 + 1];
    const scalar_t queryZ = queryPoints[i * 3 + 2];

    const int queryGridX = (int)((queryX - xmin) / aGrid);
    const int queryGridY = (int)((queryY - ymin) / aGrid);
    const int queryGridZ = (int)((queryZ - zmin) / aGrid);

    bool searchEnded = 0;  //
    int searchGridRadius = 0;
    bool alreadyFoundPos = 0;  // You need to find for another 1 gridRadius before ended.
    bool alreadyFoundNeg = 0;

    scalar_t bestSdfPos = LARGEFLOAT;
    scalar_t bestSdfNeg = LARGEFLOAT;
    scalar_t bestProjPosX = LARGEFLOAT;
    scalar_t bestProjPosY = LARGEFLOAT;
    scalar_t bestProjPosZ = LARGEFLOAT;
    scalar_t bestProjNegX = LARGEFLOAT;
    scalar_t bestProjNegY = LARGEFLOAT;
    scalar_t bestProjNegZ = LARGEFLOAT;
    scalar_t bestProjNormalPosX = LARGEFLOAT;
    scalar_t bestProjNormalPosY = LARGEFLOAT;
    scalar_t bestProjNormalPosZ = LARGEFLOAT;
    scalar_t bestProjNormalNegX = LARGEFLOAT;
    scalar_t bestProjNormalNegY = LARGEFLOAT;
    scalar_t bestProjNormalNegZ = LARGEFLOAT;
    int bestProjTrisIDPos = -1;
    int bestProjTrisIDNeg = -1;

    // DebuggingInfoExport
    scalar_t bestProjMethodPos = -1; // 1: via proj2Triangle, 2-4: via proj2Line, 5-7: via proj2Vert
    scalar_t bestProjMethodNeg = -1; // 1: via proj2Triangle, 2-4: via proj2Line, 5-7: via proj2Vert
    scalar_t breakMethod = -1; // 1: via search the whole space, 2: via searchedEnded

    while (1) {
        int queryGridXLeft = MAX(queryGridX - searchGridRadius, 0);
        int queryGridXRight = MIN(queryGridX + searchGridRadius + 1, lenGridX);
        int queryGridYLeft = MAX(queryGridY - searchGridRadius, 0);
        int queryGridYRight = MIN(queryGridY + searchGridRadius + 1, lenGridY);
        int queryGridZLeft = MAX(queryGridZ - searchGridRadius, 0);
        int queryGridZRight = MIN(queryGridZ + searchGridRadius + 1, lenGridZ);

        if (queryGridXLeft < 0 && queryGridXRight >= lenGridX &&
            queryGridYLeft < 0 && queryGridYRight >= lenGridY &&
            queryGridZLeft < 0 && queryGridZRight >= lenGridZ) {

            // DebuggingInfoExport
            breakMethod = 1;

            break;  // so that you have already searched the whole space
        }

        if (searchEnded) {

            // DebuggingInfoExport
            breakMethod = 2;

            break;
        }
        if (searchGridRadius * aGrid > bestSdfPos) alreadyFoundPos = 1;
        if (searchGridRadius * aGrid > bestSdfNeg) alreadyFoundNeg = 1;
        if (alreadyFoundPos && alreadyFoundNeg) searchEnded = 1;  // We request both of the value!!!

        for (int gridX = queryGridXLeft; gridX < queryGridXRight; gridX++) {
            for (int gridY = queryGridYLeft; gridY < queryGridYRight; gridY++) {
                for (int gridZ = queryGridZLeft; gridZ < queryGridZRight; gridZ++) {
                    if (MAX3(ABS(gridX - queryGridX),
                             ABS(gridY - queryGridY),
                             ABS(gridZ - queryGridZ)) < searchGridRadius) {
                        continue;  // which means this grid has been searched by a smaller searchRadius
                    }

                    const int gridID = gridX * lenGridY * lenGridZ + gridY * lenGridZ + gridZ;
                    const int currentGridTrisStartingIndex = gridTrisStartingIndex[b * totalGrid + gridID];
                    // to get the currentGridTrisEndingIndex
                    const int nextGridID = gridID + 1;
                    int currentGridTrisEndingIndex;
                    if (nextGridID >= totalGrid) {
                        currentGridTrisEndingIndex = nFace;
                    } else {
                        currentGridTrisEndingIndex = gridTrisStartingIndex[b * totalGrid + nextGridID];
                    }

                    /* now the alreadyFound options has been splitted into Pos and Neg
                    if (currentGridTrisEndingIndex > currentGridTrisStartingIndex) {
                        // which means you have already found any triangle at this radius.
                        // and hence you just need to do one more radius search then it would be OK.
                        // Note that so sometimes even if the mesh is not watertight, this mechanism will still
                        // give you the result as if the mesh is watertight.
                        alreadyFound = 1;
                    }
                    */

                    for (int z = currentGridTrisStartingIndex; z < currentGridTrisEndingIndex; z++) {
                        const int trisID = trisIDOrderingBuffer[b * nFace + z];
                        if (validFaceFlag[b * nFace + trisID] == 0) continue;
                        const scalar_t *faceVertNow = faceVert + b * nFace * 3 * 3 + trisID * 3 * 3;
                        const scalar_t *faceNormalNow = faceNormal + b * nFace * 3 + trisID * 3;
                        const scalar_t *faceEdgeNormalNow = faceEdgeNormal + b * nFace * 3 * 3 + trisID * 3 * 3;
                        const scalar_t *faceVertNormalNow = faceVertNormal + b * nFace * 3 * 3 + trisID * 3 * 3;
                        const scalar_t x0 = faceVertNow[0];
                        const scalar_t y0 = faceVertNow[1];
                        const scalar_t z0 = faceVertNow[2];
                        const scalar_t x1 = faceVertNow[3];
                        const scalar_t y1 = faceVertNow[4];
                        const scalar_t z1 = faceVertNow[5];
                        const scalar_t x2 = faceVertNow[6];
                        const scalar_t y2 = faceVertNow[7];
                        const scalar_t z2 = faceVertNow[8];
                        const scalar_t nxT = faceNormalNow[0];
                        const scalar_t nyT = faceNormalNow[1];
                        const scalar_t nzT = faceNormalNow[2];
                        const scalar_t nx12 = faceEdgeNormalNow[0];
                        const scalar_t ny12 = faceEdgeNormalNow[1];
                        const scalar_t nz12 = faceEdgeNormalNow[2];
                        const scalar_t nx20 = faceEdgeNormalNow[3];
                        const scalar_t ny20 = faceEdgeNormalNow[4];
                        const scalar_t nz20 = faceEdgeNormalNow[5];
                        const scalar_t nx01 = faceEdgeNormalNow[6];
                        const scalar_t ny01 = faceEdgeNormalNow[7];
                        const scalar_t nz01 = faceEdgeNormalNow[8];
                        const scalar_t nx0 = faceVertNormalNow[0];
                        const scalar_t ny0 = faceVertNormalNow[1];
                        const scalar_t nz0 = faceVertNormalNow[2];
                        const scalar_t nx1 = faceVertNormalNow[3];
                        const scalar_t ny1 = faceVertNormalNow[4];
                        const scalar_t nz1 = faceVertNormalNow[5];
                        const scalar_t nx2 = faceVertNormalNow[6];
                        const scalar_t ny2 = faceVertNormalNow[7];
                        const scalar_t nz2 = faceVertNormalNow[8];

                        // Record the things related to this triangle
                        scalar_t projXNow, projYNow, projZNow;
                        scalar_t projNxNow, projNyNow, projNzNow;
                        scalar_t d;
                        bool ifSignPos;
                        // DebuggingInfoExport
                        scalar_t projMethod;

                        // 1
                        bool ifSignPosT, ifProjInsideT;
                        scalar_t dT, projXT, projYT, projZT;
                        checkProjToTriangle(queryX, queryY, queryZ,
                                            x0, y0, z0, x1, y1, z1, x2, y2, z2,
                                            nxT, nyT, nzT,
                                            &ifSignPosT, &ifProjInsideT, &dT,
                                            &projXT, &projYT, &projZT);
                        if (ifProjInsideT) {
                            projMethod = 1;
                            ifSignPos = ifSignPosT;
                            d = dT;
                            projXNow = projXT;
                            projYNow = projYT;
                            projZNow = projZT;
                            projNxNow = nxT;
                            projNyNow = nyT;
                            projNzNow = nzT;
                        } else {
                            // 2 3 4
                            bool ifSignPos12, ifSignPos20, ifSignPos01, ifProjInside12, ifProjInside20, ifProjInside01;
                            scalar_t d12, d20, d01, projX12, projX20, projX01, projY12, projY20, projY01, projZ12, projZ20, projZ01;
                            checkProjToLine(queryX, queryY, queryZ,
                                            x1, y1, z1, x2, y2, z2,
                                            nx12, ny12, nz12,
                                            &ifSignPos12, &ifProjInside12, &d12,
                                            &projX12, &projY12, &projZ12);
                            checkProjToLine(queryX, queryY, queryZ,
                                            x2, y2, z2, x0, y0, z0,
                                            nx20, ny20, nz20,
                                            &ifSignPos20, &ifProjInside20, &d20,
                                            &projX20, &projY20, &projZ20);
                            checkProjToLine(queryX, queryY, queryZ,
                                            x0, y0, z0, x1, y1, z1,
                                            nx01, ny01, nz01,
                                            &ifSignPos01, &ifProjInside01, &d01,
                                            &projX01, &projY01, &projZ01);
                            // rot dxx if it is not valid
                            if (!ifProjInside12) d12 = LARGEFLOAT;
                            if (!ifProjInside20) d20 = LARGEFLOAT;
                            if (!ifProjInside01) d01 = LARGEFLOAT;
                            if (ifProjInside12 && (d12 <= d20 && d12 <= d01)) {
                                projMethod = 2;
                                ifSignPos = ifSignPos12;
                                d = d12;
                                projXNow = projX12;
                                projYNow = projY12;
                                projZNow = projZ12;
                                projNxNow = nx12;
                                projNyNow = ny12;
                                projNzNow = nz12;
                            } else if (ifProjInside20 && (d20 <= d01)) {
                                projMethod = 3;
                                ifSignPos = ifSignPos20;
                                d = d20;
                                projXNow = projX20;
                                projYNow = projY20;
                                projZNow = projZ20;
                                projNxNow = nx20;
                                projNyNow = ny20;
                                projNzNow = nz20;
                            } else if (ifProjInside01) {
                                projMethod = 4;
                                ifSignPos = ifSignPos01;
                                d = d01;
                                projXNow = projX01;
                                projYNow = projY01;
                                projZNow = projZ01;
                                projNxNow = nx01;
                                projNyNow = ny01;
                                projNzNow = nz01;
                            } else {
                                // 5 6 7
                                const scalar_t d0 = NORM(queryX - x0, queryY - y0, queryZ - z0);
                                const scalar_t d1 = NORM(queryX - x1, queryY - y1, queryZ - z1);
                                const scalar_t d2 = NORM(queryX - x2, queryY - y2, queryZ - z2);
                                if ((d0 <= d1) && (d0 <= d2)) {
                                    projMethod = 5;
                                    ifSignPos = SGN(DOT(queryX - x0, queryY - y0, queryZ - z0, nx0, ny0, nz0));
                                    d = d0;
                                    projXNow = x0;
                                    projYNow = y0;
                                    projZNow = z0;
                                    projNxNow = nx0;
                                    projNyNow = ny0;
                                    projNzNow = nz0;
                                } else if (d1 <= d2) {
                                    projMethod = 6;
                                    ifSignPos = SGN(DOT(queryX - x1, queryY - y1, queryZ - z1, nx1, ny1, nz1));
                                    d = d1;
                                    projXNow = x1;
                                    projYNow = y1;
                                    projZNow = z1;
                                    projNxNow = nx1;
                                    projNyNow = ny1;
                                    projNzNow = nz1;
                                } else {
                                    projMethod = 7;
                                    ifSignPos = SGN(DOT(queryX - x2, queryY - y2, queryZ - z2, nx2, ny2, nz2));
                                    d = d2;
                                    projXNow = x2;
                                    projYNow = y2;
                                    projZNow = z2;
                                    projNxNow = nx2;
                                    projNyNow = ny2;
                                    projNzNow = nz2;
                                }
                            }
                        }

                        if (ifSignPos) {  // query point is outside
                            if (d < bestSdfPos) {
                                // alreadyFoundPos = 1;
                                bestSdfPos = d;
                                bestProjPosX = projXNow;
                                bestProjPosY = projYNow;
                                bestProjPosZ = projZNow;
                                bestProjNormalPosX = projNxNow;
                                bestProjNormalPosY = projNyNow;
                                bestProjNormalPosZ = projNzNow;
                                bestProjTrisIDPos = trisID;

                                // DebuggingInfoExport
                                bestProjMethodPos = projMethod;
                            }
                        } else {  // query point is inside
                            if (d < bestSdfNeg) {
                                // alreadyFoundNeg = 1;
                                bestSdfNeg = d;
                                bestProjNegX = projXNow;
                                bestProjNegY = projYNow;
                                bestProjNegZ = projZNow;
                                bestProjNormalNegX = projNxNow;
                                bestProjNormalNegY = projNyNow;
                                bestProjNormalNegZ = projNzNow;
                                bestProjTrisIDNeg = trisID;

                                // DebuggingInfoExport
                                bestProjMethodNeg = projMethod;
                            }
                        }
                        // Done.

                    }
                }
            }
        }
        searchGridRadius++;
    }

    sdfPos[i] = bestSdfPos;
    sdfNeg[i] = bestSdfNeg;
    projPointPos[i * 3 + 0] = bestProjPosX;
    projPointPos[i * 3 + 1] = bestProjPosY;
    projPointPos[i * 3 + 2] = bestProjPosZ;
    projPointNeg[i * 3 + 0] = bestProjNegX;
    projPointNeg[i * 3 + 1] = bestProjNegY;
    projPointNeg[i * 3 + 2] = bestProjNegZ;
    projPointNormalPos[i * 3 + 0] = bestProjNormalPosX;
    projPointNormalPos[i * 3 + 1] = bestProjNormalPosY;
    projPointNormalPos[i * 3 + 2] = bestProjNormalPosZ;
    projPointNormalNeg[i * 3 + 0] = bestProjNormalNegX;
    projPointNormalNeg[i * 3 + 1] = bestProjNormalNegY;
    projPointNormalNeg[i * 3 + 2] = bestProjNormalNegZ;
    projTrisIDPos[i] = bestProjTrisIDPos;
    projTrisIDNeg[i] = bestProjTrisIDNeg;

    // DebuggingInfoExport
    if (dimDebug >= 20) {
        int basePos, baseNeg;  // So all the used side (pos/neg) are always stored in 0-9.
        debuggingInfoExport[i * dimDebug + 10] = searchGridRadius;
        if (bestSdfPos <= bestSdfNeg) {
            debuggingInfoExport[i * dimDebug + 0] = 1;
            basePos = 0;
            baseNeg = 10;
        } else {
            debuggingInfoExport[i * dimDebug + 0] = -1;
            basePos = 10;
            baseNeg = 0;
        }
        debuggingInfoExport[i * dimDebug + basePos + 1] = bestProjMethodPos;
        debuggingInfoExport[i * dimDebug + baseNeg + 1] = bestProjMethodNeg;
        debuggingInfoExport[i * dimDebug + basePos + 2] = bestSdfPos;
        debuggingInfoExport[i * dimDebug + baseNeg + 2] = bestSdfNeg;
        debuggingInfoExport[i * dimDebug + basePos + 3] = bestProjTrisIDPos;
        debuggingInfoExport[i * dimDebug + baseNeg + 3] = bestProjTrisIDNeg;
        // debuggingInfoExport[i * dimDebug + 5] = breakMethod;
    }
}

/*
template<typename scalar_t>
__global__ void judge_if_face_in_through_cuda_kernel(
        const scalar_t* faceVert,
        uint8_t* queryGridIfFaceInThrough,
        int m,
        int nFace,
        int totalQueryGrid,
        int lenQueryGridX,
        int lenQueryGridY,
        int lenQueryGridZ,
        float xmin,
        float xmax,
        float ymin,
        float ymax,
        float zmin,
        float zmax,
        float aQueryGrid) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * nFace) return;
    const int b = i / nFace;
    const int f = i - b * nFace;
    const scalar_t x0 = faceVert[i * 3 * 3 + 0];
    const scalar_t y0 = faceVert[i * 3 * 3 + 1];
    const scalar_t z0 = faceVert[i * 3 * 3 + 2];
    const scalar_t x1 = faceVert[i * 3 * 3 + 3];
    const scalar_t y1 = faceVert[i * 3 * 3 + 4];
    const scalar_t z1 = faceVert[i * 3 * 3 + 5];
    const scalar_t x2 = faceVert[i * 3 * 3 + 6];
    const scalar_t y2 = faceVert[i * 3 * 3 + 7];
    const scalar_t z2 = faceVert[i * 3 * 3 + 8];

    const scalar_t edgeLength0 = NORM(x2 - x1, y2 - y1, z2 - z1);
    const scalar_t edgeLength1 = NORM(x0 - x2, y0 - y2, z0 - z2);
    const scalar_t edgeLength2 = NORM(x1 - x0, y1 - y0, z1 - z0);
    const scalar_t edgeLengthMax = MAX3(edgeLength0, edgeLength1, edgeLength2);

    const int splits = (int)ceil((float)edgeLengthMax / (float)aQueryGrid * 2.);
    for (int alphaInt = 0; alphaInt <= splits; alphaInt++) {
        const scalar_t alpha = (float)alphaInt / (float)splits;
        for (int betaInt = 0; betaInt <= splits; betaInt++) {
            const scalar_t beta = (float)betaInt / (float)splits;
            const scalar_t gamma = 1. - alpha - beta;
            const scalar_t xNow = x0 * alpha + x1 * beta + x2 * gamma;
            const scalar_t yNow = y0 * alpha + y1 * beta + y2 * gamma;
            const scalar_t zNow = z0 * alpha + z1 * beta + z2 * gamma;
            const int qeuryGridX = (int)(xNow / aQueryGrid);
            const int queryGridY = (int)(yNow / aQueryGrid);
            const int queryGridZ = (int)(zNow / aQueryGrid);
            const int queryGridID = queryGridX * lenQueryGridY * lenQueryGridZ + queryGridY * lenQueryGridZ + queryGridZ;

            queryGridIfFaceInThrough[b * lenQueryGridX * lenQueryGridY * lenQueryGridZ + queryGridID] = 1;
        }
    }

}
*/

}

std::vector<at::Tensor> sdf_tris_search_cuda(
        at::Tensor queryPoints,
        at::Tensor trisIDOrderingBuffer,
        at::Tensor gridTrisStartingIndex,
        at::Tensor faceVert,
        at::Tensor faceNormal,
        at::Tensor faceEdgeNormal,
        at::Tensor faceVertNormal,
        at::Tensor validFaceFlag,
        at::Tensor sdfPos,
        at::Tensor sdfNeg,
        at::Tensor projPointPos,
        at::Tensor projPointNeg,
        at::Tensor projPointNormalPos,
        at::Tensor projPointNormalNeg,
        at::Tensor projTrisIDPos,
        at::Tensor projTrisIDNeg,
        at::Tensor debuggingInfoExport,
        float xmin,
        float xmax,
        float ymin,
        float ymax,
        float zmin,
        float zmax,
        float aGrid) {

    /*
    Inputs:
        queryPoints: m * nQuery * 3
        trisIDOrderingBuffer: m * nFace
        gridTrisStartingIndex: m * totalGrid
        faceVert: m * nFace * 3(012) * 3(xyz)
        faceNormal: m * nFace * 3(nxnynz)
        faceEdgeNormal: m * nFace * 3(122001) * 3(nxnynz)
        faceVertNormal: m * nFace * 3(012) * 3(nxnynz)
        validFaceFlag (int32): m * nFace

    Outputs:
        sdfPos: m * nQuery  // positive is outside, and negative is inside
        sdfNeg
        projPointPos: m * nQuery * 3
        projPointNeg
        projPointNormalPos: m * nQuery * 3
        projPointNormalNeg:
        projTrisIDPos: m * nQuery
        projTrisIDNeg
        debuggingInfoExport: m * nQuery * dimDebug
    */

    CHECK_INPUT(queryPoints);
    TORCH_CHECK(queryPoints.dim() == 3, "sdf_tris_search_cuda: queryPoints.dim() wrong");
    const auto m = queryPoints.size(0);
    const auto nQuery = queryPoints.size(1);
    TORCH_CHECK(queryPoints.size(2) == 3, "sdf_tris_search_cuda: queryPoints.size(2) wrong");

    CHECK_INPUT(trisIDOrderingBuffer);
    TORCH_CHECK(trisIDOrderingBuffer.dim() == 2, "sdf_tris_search_cuda: trisIDOrderingBuffer.dim() wrong");
    TORCH_CHECK(trisIDOrderingBuffer.size(0) == m, "sdf_tris_search_cuda: trisIDOrderingBuffer.size(0) wrong");
    const auto nFace = trisIDOrderingBuffer.size(1);

    CHECK_INPUT(gridTrisStartingIndex);
    TORCH_CHECK(gridTrisStartingIndex.dim() == 2, "sdf_tris_search_cuda: gridTrisStartingIndex.dim() wrong");
    TORCH_CHECK(gridTrisStartingIndex.size(0) == m, "sdf_tris_search_cuda: gridTrisStartingIndex.size(0) wrong");
    const auto totalGrid = gridTrisStartingIndex.size(1);

    CHECK_INPUT(faceVert);
    TORCH_CHECK(faceVert.dim() == 4 && faceVert.size(0) == m && faceVert.size(1) == nFace &&
             faceVert.size(2) == 3 && faceVert.size(3) == 3,
             "sdf_tris_search_cuda: faceVert shape wrong");

    CHECK_INPUT(faceNormal);
    TORCH_CHECK(faceNormal.dim() == 3 && faceNormal.size(0) == m && faceNormal.size(2) == 3,
             "sdf_tris_search_cuda: faceNormal shape wrong");

    CHECK_INPUT(faceEdgeNormal);
    TORCH_CHECK(faceEdgeNormal.dim() == 4 && faceEdgeNormal.size(0) == m && faceEdgeNormal.size(1) == nFace &&
             faceEdgeNormal.size(2) == 3 && faceEdgeNormal.size(3) == 3, "sdf_tris_search_cuda: faceEdgeNormal shape wrong");

    CHECK_INPUT(faceVertNormal);
    TORCH_CHECK(faceVertNormal.dim() == 4 && faceVertNormal.size(0) == m && faceVertNormal.size(1) == nFace &&
             faceVertNormal.size(2) == 3 && faceVertNormal.size(3) == 3,
             "sdf_tris_search_cuda: vertNormal shape wrong");
    CHECK_INPUT(validFaceFlag);
    TORCH_CHECK(validFaceFlag.dim() == 2 && validFaceFlag.size(0) == m && validFaceFlag.size(1) == nFace,
             "validFaceFlag shape wrong");

    CHECK_INPUT(sdfPos);
    TORCH_CHECK(sdfPos.dim() == 2 && sdfPos.size(0) == m && sdfPos.size(1) == nQuery, "sdf_tris_search_cuda: sdfPos shape wrong");
    CHECK_INPUT(sdfNeg);
    TORCH_CHECK(sdfNeg.dim() == 2 && sdfNeg.size(0) == m && sdfNeg.size(1) == nQuery, "sdf_tris_search_cuda: sdfNeg shape wrong");
    CHECK_INPUT(projPointPos);
    TORCH_CHECK(projPointPos.dim() == 3 && projPointPos.size(0) == m && projPointPos.size(1) == nQuery && projPointPos.size(2) == 3,
             "sdf_tris_search_cuda: projPointPos shape wrong");
    CHECK_INPUT(projPointNeg);
    TORCH_CHECK(projPointNeg.dim() == 3 && projPointNeg.size(0) == m && projPointNeg.size(1) == nQuery && projPointNeg.size(2) == 3,
             "sdf_tris_search_cuda: projPointNeg shape wrong");
    CHECK_INPUT(projPointNormalPos);
    TORCH_CHECK(projPointNormalPos.dim() == 3 && projPointNormalPos.size(0) == m && projPointNormalPos.size(1) == nQuery &&
             projPointNormalPos.size(2) == 3,
             "sdf_tris_search_cuda: projPointNormalPos shape wrong");
    CHECK_INPUT(projPointNormalNeg);
    TORCH_CHECK(projPointNormalNeg.dim() == 3 && projPointNormalNeg.size(0) == m && projPointNormalNeg.size(1) == nQuery &&
             projPointNormalNeg.size(2) == 3,
             "sdf_tris_search_cuda: projPointNormalNeg shape wrong");
    CHECK_INPUT(projTrisIDPos);
    TORCH_CHECK(projTrisIDPos.dim() == 2 && projTrisIDPos.size(0) == m && projTrisIDPos.size(1) == nQuery,
             "sdf_tris_search_cuda: projTrisIDPos shape wrong");
    CHECK_INPUT(projTrisIDNeg);
    TORCH_CHECK(projTrisIDNeg.dim() == 2 && projTrisIDNeg.size(0) == m && projTrisIDNeg.size(1) == nQuery,
             "sdf_tris_search_cuda: projTrisIDNeg shape wrong");

    CHECK_INPUT(debuggingInfoExport);
    TORCH_CHECK(debuggingInfoExport.dim() == 3 && debuggingInfoExport.size(0) == m &&
             debuggingInfoExport.size(1) == nQuery,
             "sdf_tris_search_cuda: debuggingInfoExport shape wrong");
    const auto dimDebug = debuggingInfoExport.size(2);

    const auto lenGridX = (int)ceil((xmax - xmin) / aGrid);
    const auto lenGridY = (int)ceil((ymax - ymin) / aGrid);
    const auto lenGridZ = (int)ceil((zmax - zmin) / aGrid);
    TORCH_CHECK(lenGridX * lenGridY * lenGridZ == totalGrid,
        "sdf_tris_search_cuda: lenGridX * lenGridY * lenGridZ == totalGrid does not hold");

    const int threads = 512;
    const dim3 blocks((m * nQuery - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(queryPoints.type(), "sdf_tris_search_cuda", ([&] {
        sdf_tris_search_cuda_kernel<scalar_t><<<blocks, threads>>>(
            queryPoints.data<scalar_t>(),
            trisIDOrderingBuffer.data<int32_t>(),
            gridTrisStartingIndex.data<int32_t>(),
            faceVert.data<scalar_t>(),
            faceNormal.data<scalar_t>(),
            faceEdgeNormal.data<scalar_t>(),
            faceVertNormal.data<scalar_t>(),
            validFaceFlag.data<int32_t>(),
            sdfPos.data<scalar_t>(),
            sdfNeg.data<scalar_t>(),
            projPointPos.data<scalar_t>(),
            projPointNeg.data<scalar_t>(),
            projPointNormalPos.data<scalar_t>(),
            projPointNormalNeg.data<scalar_t>(),
            projTrisIDPos.data<int32_t>(),
            projTrisIDNeg.data<int32_t>(),
            debuggingInfoExport.data<scalar_t>(),
            m,
            nQuery,
            nFace,
            totalGrid,
            lenGridX,
            lenGridY,
            lenGridZ,
            dimDebug,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            aGrid
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in sdf_tris_search_cuda_kernel: %s\n", cudaGetErrorString(err));
    }
    return {sdfPos, sdfNeg, projPointPos, projPointNeg, projPointNormalPos, projPointNormalNeg,
            projTrisIDPos, projTrisIDNeg, debuggingInfoExport};
}

std::vector<at::Tensor> point_id_rearrange_cpu(
        at::Tensor pointGridXYZ,
        at::Tensor pointIDOrderingBuffer,
        at::Tensor gridPointStartingIndex) {

    /*
    Inputs:
        pointGridXYZ: m * nPoint * 3(xyz)

    Outputs:
        pointIDOrderingBuffer: m * nPoint
        gridPointStartingIndex: m * lenGridX * lenGridY * lenGridZ
    */
    CHECK_CPUINPUT(pointGridXYZ);
    TORCH_CHECK(pointGridXYZ.dim() == 3, "point_id_rearrange_cpu: pointGridXYZ.(dim) wrong");
    const auto m = pointGridXYZ.size(0);
    const auto nPoint = pointGridXYZ.size(1);
    TORCH_CHECK(pointGridXYZ.size(2) == 3, "point_id_rearrange_cpu: pointGridXYZ.size(2) wrong");

    CHECK_CPUINPUT(pointIDOrderingBuffer);
    TORCH_CHECK(pointIDOrderingBuffer.dim() == 2, "point_id_rearrange_cpu: pointIDOrderingBuffer.dim() wrong");
    TORCH_CHECK(pointIDOrderingBuffer.size(0) == m, "point_id_rearrange_cpu: pointIDOrderingBuffer.size(0) wrong");
    TORCH_CHECK(pointIDOrderingBuffer.size(1) == nPoint, "point_id_rearrange_cpu: pointIDOrderingBuffer.size(1) wrong");

    CHECK_CPUINPUT(gridPointStartingIndex);
    TORCH_CHECK(gridPointStartingIndex.dim() == 4);
    TORCH_CHECK(gridPointStartingIndex.size(0) == m);
    const auto lenGridX = gridPointStartingIndex.size(1);
    const auto lenGridY = gridPointStartingIndex.size(2);
    const auto lenGridZ = gridPointStartingIndex.size(3);
    const auto totalGrid = lenGridX * lenGridY * lenGridZ;

    for (int b = 0; b < m; b++) {
        const int32_t *_pointGridXYZ = pointGridXYZ.data<int32_t>() + b * nPoint * 3;
        int32_t *_pointIDOrderingBuffer = pointIDOrderingBuffer.data<int32_t>() + b * nPoint;
        int32_t *_gridPointStartingIndex = gridPointStartingIndex.data<int32_t>() + b * lenGridX * lenGridY * lenGridZ;

        std::vector<std::vector<int32_t>> gridStack(totalGrid);
        for (int point = 0; point < nPoint; point++) {
            const int xGrid = _pointGridXYZ[point * 3 + 0];
            const int yGrid = _pointGridXYZ[point * 3 + 1];
            const int zGrid = _pointGridXYZ[point * 3 + 2];
            TORCH_CHECK(0 <= xGrid);
            TORCH_CHECK(0 <= yGrid);
            TORCH_CHECK(0 <= zGrid);
            TORCH_CHECK(xGrid < lenGridX);
            TORCH_CHECK(yGrid < lenGridY);
            TORCH_CHECK(zGrid < lenGridZ);
            const int grid = xGrid * lenGridY * lenGridZ + yGrid * lenGridZ + zGrid;
            gridStack[grid].push_back(point);
        }
        int count = 0;
        for (int xGrid = 0; xGrid < lenGridX; xGrid++) {
            for (int yGrid = 0; yGrid < lenGridY; yGrid++) {
                for (int zGrid = 0; zGrid < lenGridZ; zGrid++) {
                    const int grid = xGrid * lenGridY * lenGridZ + yGrid * lenGridZ + zGrid;
                    _gridPointStartingIndex[grid] = count;
                    for (int k = 0; k < gridStack[grid].size(); k++) {
                        _pointIDOrderingBuffer[count] = gridStack[grid][k];
                        count++;
                    }
                }
            }
        }
    }

    return {pointIDOrderingBuffer, gridPointStartingIndex};
}

// at::Tensor judge_if_face_in_through(at::Tensor faceVert,
//                                at::Tensor queryGridIfFaceInThrough,
//                                float xmin,
//                                float xmax,
//                                float ymin,
//                                float ymax,
//                                float zmin,
//                                float zmax,
//                                float aQueryGrid) {
//
//     /*
//         Inputs:
//             faceVert: m * nFace * 3(012) * 3(xyz)
//
//         Outputs:
//             queryGirdIfFaceInThrough: m * lenQueryGridX * lenQueryGridY * lenQueryGridZ
//     */
//
//     CHECK_INPUT(faceVert);
//     const auto m = faceVert.size(0);
//     const auto nFace = faceVert.size(1);
//     TORCH_CHECK(faceVert.dim() == 4 && faceVert.size(2) == 3 && faceVert.size(3) == 3,
//              "faceVert shape wrong");
//
//     const auto lenQueryGridX = (int)ceil((xmax - xmin) / aQueryGrid);
//     const auto lenQueryGridY = (int)ceil((ymax - ymin) / aQueryGrid);
//     const auto lenQueryGridZ = (int)ceil((zmax - zmin) / aQueryGrid);
//     const auto totalQueryGrid = lenQueryGridX * lenQueryGridY * lenQueryGridZ;
//
//     CHECK_INPUT(queryGridIfFaceInThrough);
//     // You need to check outside that all of its values are 0.
//     // Since you create it in python so that would be the same as "check".
//     TORCH_CHECK(queryGridIfFaceInThrough.dim() == 4 &&
//              queryGridIfFaceInThrough.size(0) == m &&
//              queryGridIfFaceInThrough.size(1) == lenQueryGridX &&
//              queryGridIfFaceInThrough.size(2) == lenQueryGridY &&
//              queryGridIfFaceInThrough.size(3) == lenQueryGridZ,
//              "queryGridIfFaceInThrough shape wrong");
//
//     const int threads = 512;
//     const dim3 blocks((m * nFace - 1) / threads + 1);
//
//     AT_DISPATCH_FLOATING_TYPES(faceVert.type(), "judge_if_face_in_through_cuda", ([&] {
//         judge_if_face_in_through_cuda_kernel<scalar_t><<<blocks, threads>>>(
//             faceVert.data<scalar_t>(),
//             queryGridIfFaceInThrough.data<uint8_t>(),
//             m,
//             nFace,
//             totalQueryGrid,
//             lenQueryGridX,
//             lenQueryGridY,
//             lenQueryGridZ,
//             xmin,
//             xmax,
//             ymin,
//             ymax,
//             zmin,
//             zmax,
//             aQueryGrid
//         );
//     }));
//
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("Error in judge_if_face_in_through_cuda: %s\n", cudaGetErrorString(err));
//     }
//     return queryGridIfFaceInThrough;
// }


