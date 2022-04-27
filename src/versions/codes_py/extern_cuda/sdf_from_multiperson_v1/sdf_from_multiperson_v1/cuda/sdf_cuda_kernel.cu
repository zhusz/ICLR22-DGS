#include <ATen/ATen.h>

#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2];

#define LOC(b, c0, c1, c2, C0, C1, C2)  ((((b) * (C0) + (c0)) * (C1) + (c1)) * (C2) + (c2))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SIGN(x) (((x) >= 0) ? (1) : (-1))
#define ABS(x) ((x) >= 0 ? (x) : (-(x)))

namespace {

template<typename scalar_t>
static __inline__ __device__ scalar_t mag2(const scalar_t* x) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * x[i];
    }
    return l;
}

template<typename scalar_t>
static __inline__ __device__ scalar_t norm(const scalar_t* x) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * x[i];
    }
    return sqrt(l);
}

template<typename scalar_t>
static __inline__ __device__ scalar_t dist2(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    scalar_t diff;
    for (int i=0; i<3; ++i) {
        diff = x[i] - y[i];
        l += diff * diff;
    }
    return l;
}

template<typename scalar_t>
static __inline__ __device__ scalar_t dist(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    scalar_t diff;
    for (int i=0; i<3; ++i) {
        diff = x[i] - y[i];
        l += diff * diff;
    }
    return sqrt(l);
}


template<typename scalar_t>
static __inline__ __device__ scalar_t dot(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * y[i];
    }
    return l;
}


// find distance x0 is from segment x1-x2
template<typename scalar_t>
static __inline__ __device__ scalar_t point_segment_distance(const scalar_t* x0, const scalar_t* x1, const scalar_t* x2, scalar_t* r)
{
    scalar_t dx[3] = {x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2]};
    scalar_t m2 = mag2(dx);
    // find parameter value of closest point on segment
    // scalar_t s12= (scalar_t) (dot(x2-x0, dx)/m2);
    scalar_t s12 = (scalar_t) (dot(x2, dx) - dot(x0, dx)) / m2;
    if (s12 < 0){
       s12 = 0;
    }
    else if (s12 > 1){
       s12 = 1;
    }
    for (int i=0; i < 3; ++i) {
        r[i] = s12*x1[i] + (1-s12) * x2[i];
    }
    // and find the distance
    return dist(x0, r);
}

/* the original jgt code */
template<typename scalar_t>
static __inline__ __device__ int intersect_triangle(
               const scalar_t* orig, const scalar_t* dir,
		       const scalar_t* vert0, const scalar_t* vert1,
               const scalar_t* vert2, scalar_t* t, scalar_t *u, scalar_t *v) {

    scalar_t edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    scalar_t det,inv_det;

    /* find vectors for two edges sharing vert0 */
    SUB(edge1, vert1, vert0);
    SUB(edge2, vert2, vert0);

    /* begin calculating determinant - also used to calculate U parameter */
    CROSS(pvec, dir, edge2);  // pvec is h

    /* if determinant is near zero, ray lies in plane of triangle */
    det = DOT(edge1, pvec);  // det is a

    if (det > -EPSILON && det < EPSILON)
        return 0;
    inv_det = 1.0 / det;  // inv_det is f

    /* calculate distance from vert0 to ray origin */
    SUB(tvec, orig, vert0);  // tvec is s

    /* calculate U parameter and test bounds */
    *u = DOT(tvec, pvec) * inv_det;
    if (*u < 0.0 || *u > 1.0)
        return 0;

    /* prepare to test V parameter */
    CROSS(qvec, tvec, edge1);  // qvec is q

    /* calculate V parameter and test bounds */
    *v = DOT(dir, qvec) * inv_det;
    if (*v < 0.0 || (*u + *v) > 1.0)
        return 0;

    /* calculate t, ray intersects triangle */
    *t = DOT(edge2, qvec) * inv_det;

    return 1;
}

template<typename scalar_t>
static __inline__ __device__ int triangle_ray_intersection(const scalar_t* origin, const scalar_t* dest,
    const scalar_t* v1, const scalar_t* v2, const scalar_t* v3, scalar_t* t) {

    scalar_t _dir[3] = {dest[0] - origin[0], dest[1] - origin[1], dest[2] - origin[2]};

    // t is the distance, u and v are barycentric coordinates
    // http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/code/raytri_tam.pdf
    scalar_t u, v;
    return intersect_triangle(origin, _dir, v1, v2, v3, t, &u, &v);
}



// find distance x0 is from triangle x1-x2-x3
template<typename scalar_t>
// static scalar_t point_triangle_distance(const Vec3f &&x0, const Vec3f &&x1, const Vec3f &&x2, const Vec3f &&x3)
static __inline__ __device__ scalar_t point_triangle_distance(const scalar_t* x0, const scalar_t* x1, const scalar_t* x2, const scalar_t* x3, scalar_t* r) {
   // first find barycentric coordinates of closest point on infinite plane
    scalar_t x13[3];
    scalar_t x23[3];
    scalar_t x03[3];
    for (int i=0; i<3; ++i) {
        x13[i] = x1[i] - x3[i];
        x23[i] = x2[i] - x3[i];
        x03[i] = x0[i] - x3[i];
    }
    scalar_t m13 = mag2(x13);
    scalar_t m23 = mag2(x23);
    scalar_t m33 = mag2(x03);
    scalar_t d = dot(x13, x23);
    scalar_t invdet=1.f/max(m13*m23-d*d,1e-30f);
    scalar_t a = dot(x13, x03);
    scalar_t b = dot(x23, x03);
    // the barycentric coordinates themselves
    scalar_t w23=invdet*(m23*a-d*b);
    scalar_t w31=invdet*(m13*b-d*a);
    scalar_t w12=1-w23-w31;

    if (w23>=0 && w31>=0 && w12>=0){ // if we're inside the triangle
        for (int i=0; i<3; ++i) {
            r[i] = w23*x1[i] + w31*x2[i]+w12*x3[i];
        }
        return dist(x0, r);
    }
    else { // we have to clamp to one of the edges
        scalar_t r1[3] = {0,0,0};
        scalar_t r2[3] = {0,0,0};
        if (w23 > 0) {// this rules out edge 2-3 for us
            scalar_t d1 = point_segment_distance(x0,x1,x2,r1);
            scalar_t d2 = point_segment_distance(x0,x1,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else if (w31 > 0) {// this rules out edge 1-3
            scalar_t d1 = point_segment_distance(x0,x1,x2,r1);
            scalar_t d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else {// w12 must be >0, ruling out edge 1-2
            scalar_t d1 = point_segment_distance(x0,x1,x3,r1);
            scalar_t d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
    }
}


template<typename scalar_t>
__global__ void sdf_cuda_kernel(
        const scalar_t* queryPoint,
        const scalar_t* vert,
        const int32_t* face,
        const scalar_t* referencePoints,
        scalar_t* __restrict__ sdf,
        scalar_t* __restrict__ projPoint,
        int32_t* __restrict__ cudaIntBuffer,
        int B,
        int Q,
        int nVert,
        int nFace,
        int R,
        int Dint
    ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Q) {
        return;
    }

    const int b = i / Q;
    // const int q = i % Q;

    for (int r = 0; r < R; r++) cudaIntBuffer[i * Dint + r] = 0;
    scalar_t min_distance = MAXFLOAT;
    scalar_t best_proj_x = 0.;
    scalar_t best_proj_y = 0.;
    scalar_t best_proj_z = 0.;
    for (int f = 0; f < nFace; f++) {
        const int v0 = face[b * (nFace * 3) + f * 3 + 0];
        const int v1 = face[b * (nFace * 3) + f * 3 + 1];
        const int v2 = face[b * (nFace * 3) + f * 3 + 2];
        const scalar_t* queryPointNow = queryPoint + i * 3;
        const scalar_t* vertNow0 = vert + b * (nVert * 3) + v0 * 3;
        const scalar_t* vertNow1 = vert + b * (nVert * 3) + v1 * 3;
        const scalar_t* vertNow2 = vert + b * (nVert * 3) + v2 * 3;
        point_triangle_distance(
            queryPointNow,
            vertNow0,
            vertNow1,
            vertNow2,
            projPoint + i * 3
        );
        scalar_t distance = dist(queryPointNow, projPoint + i * 3);
        // bool flagYes = 0;
        if (distance < min_distance) {
            min_distance = distance;
            best_proj_x = projPoint[i * 3 + 0];
            best_proj_y = projPoint[i * 3 + 1];
            best_proj_z = projPoint[i * 3 + 2];
            // flagYes = 1;
        }
        // Möller–Trumbore intersection algorithm

        for (int r = 0; r < R; r++) {
            bool intersect = triangle_ray_intersection(
                referencePoints + b * (R * 3) + r * 3,
                queryPointNow,
                vertNow0,
                vertNow1,
                vertNow2,
                sdf + i
            );
            if (intersect && (sdf[i] >= 0) && (sdf[i] <= 1)) {  // line segment should be between [0, 1]
                // num_intersect++;
                cudaIntBuffer[i * Dint + r] += 1;
            }
        }
        // if ((D >= 3) && flagYes) {  // only the last hitted triangle's result remains
        //     debuggingInfo[i * D + 1] = distance;
        //     debuggingInfo[i * D + 2] = sdf[i];
        // }
        // if ((D >= 6) && flagYes) {  // only the last hitted triangle's result remains
        //     for (int z = 0; z < 3; z++) {
        //         debuggingInfo[i * D + 3 + z] =
        //             referenceSinglePoint[b * 3 + z] +
        //             sdf[i] * (queryPointNow[z] - referenceSinglePoint[b * 3 + z]);
        //     }
        // }
    }
    // vote to win
    int winning_thre = (R + 1) / 2;
    int count = 0;
    for (int r = 0; r < R; r++) {
        if (cudaIntBuffer[i * Dint + r] % 2 == 1) {
            count++;
        }
    }
    if (count >= winning_thre) {
        min_distance *= -1;  // the flip wins!
    }
    // if (num_intersect % 2 == 1) {
    //    min_distance *= -1;
    // }
    sdf[i] = min_distance;
    projPoint[i * 3 + 0] = best_proj_x;
    projPoint[i * 3 + 1] = best_proj_y;
    projPoint[i * 3 + 2] = best_proj_z;
}


template<typename scalar_t>
__global__ void udf_cuda_kernel(
        const scalar_t* queryPoint,
        const scalar_t* vert,
        const int32_t* face,
        scalar_t* __restrict__ udf,
        scalar_t* __restrict__ projPoint,
        int B,
        int Q,
        int nVert,
        int nFace) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Q) {
        return;
    }

    const int b = i / Q;
    // const int q = i % Q;

    scalar_t min_distance = MAXFLOAT;
    scalar_t best_proj_x = 0.;
    scalar_t best_proj_y = 0.;
    scalar_t best_proj_z = 0.;
    for (int f = 0; f < nFace; f++) {
        const int v0 = face[b * (nFace * 3) + f * 3 + 0];
        const int v1 = face[b * (nFace * 3) + f * 3 + 1];
        const int v2 = face[b * (nFace * 3) + f * 3 + 2];
        const scalar_t* queryPointNow = queryPoint + i * 3;
        const scalar_t* vertNow0 = vert + b * (nVert * 3) + v0 * 3;
        const scalar_t* vertNow1 = vert + b * (nVert * 3) + v1 * 3;
        const scalar_t* vertNow2 = vert + b * (nVert * 3) + v2 * 3;
        point_triangle_distance(
            queryPointNow,
            vertNow0,
            vertNow1,
            vertNow2,
            projPoint + i * 3
        );
        scalar_t distance = dist(queryPointNow, projPoint + i * 3);
        if (distance < min_distance) {
            min_distance = distance;
            best_proj_x = projPoint[i * 3 + 0];
            best_proj_y = projPoint[i * 3 + 1];
            best_proj_z = projPoint[i * 3 + 2];
        }
    }
    udf[i] = min_distance;
    projPoint[i * 3 + 0] = best_proj_x;
    projPoint[i * 3 + 1] = best_proj_y;
    projPoint[i * 3 + 2] = best_proj_z;
}

template<typename scalar_t>
__global__ void sign_value_matching_cuda_kernel(
        const scalar_t* mismatchedSdf,
        int radius,
        int B, int C0, int C1, int C2,
        scalar_t* __restrict__ matchedSdf
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= B * C0 * C1 * C2) return;

    const int b = g / (C0 * C1 * C2);
    const int c0 = (g - b * C0 * C1 * C2) / (C1 * C2);
    const int c1 = (g - (b * C0 + c0) * C1 * C2) / C2;
    const int c2 = g % C2;

    float score0;
    int countChangeSign0 = 0;
    const int signLeft0 = SIGN(mismatchedSdf[LOC(b, MAX(c0 - radius, 0), c1, c2, C0, C1, C2)]);
    for (int k = MAX(c0 - radius, 0) + 1; k < MIN(c0 + radius + 1, C0); k++) {
        if (SIGN(mismatchedSdf[LOC(b, k, c1, c2, C0, C1, C2)]) !=
            SIGN(mismatchedSdf[LOC(b, k - 1, c1, c2, C0, C1, C2)])) countChangeSign0++;
    }
    if (countChangeSign0 == 1) {
        score0 = mismatchedSdf[LOC(b, MIN(c0 + radius + 1, C0) - 1, c1, c2, C0, C1, C2)] -
                 mismatchedSdf[LOC(b, MAX(c0 - radius, 0), c1, c2, C0, C1, C2)];
        score0 = ABS(score0);
    } else {
        score0 = 0.;
    }

    float score1;
    int countChangeSign1 = 0;
    const int signLeft1 = SIGN(mismatchedSdf[LOC(b, c0, MAX(c1 - radius, 0), c2, C0, C1, C2)]);
    for (int k = MAX(c1 - radius, 0) + 1; k < MIN(c1 + radius + 1, C1); k++) {
        if (SIGN(mismatchedSdf[LOC(b, c0, k, c2, C0, C1, C2)]) !=
            SIGN(mismatchedSdf[LOC(b, c0, k - 1, c2, C0, C1, C2)])) countChangeSign1++;
    }
    if (countChangeSign1 == 1) {
        score1 = mismatchedSdf[LOC(b, c0, MIN(c1 + radius + 1, C1) - 1, c2, C0, C1, C2)] -
                 mismatchedSdf[LOC(b, c0, MAX(c1 - radius, 0), c2, C0, C1, C2)];
        score1 = ABS(score1);
    } else {
        score1 = 0.;
    }

    float score2;
    int countChangeSign2 = 0;
    const int signLeft2 = SIGN(mismatchedSdf[LOC(b, c0, c1, MAX(c2 - radius, 0), C0, C1, C2)]);
    for (int k = MAX(c2 - radius, 0) + 1; k < MIN(c2 + radius + 1, C2); k++) {
        if (SIGN(mismatchedSdf[LOC(b, c0, c1, k, C0, C1, C2)]) !=
            SIGN(mismatchedSdf[LOC(b, c0, c1, k - 1, C0, C1, C2)])) countChangeSign2++;
    }
    if (countChangeSign2 == 1) {
        score2 = mismatchedSdf[LOC(b, c0, c1, MIN(c2 + radius + 1, C2) -1, C0, C1, C2)] -
                 mismatchedSdf[LOC(b, c0, c1, MAX(c2 - radius, 0), C0, C1, C2)];
        score2 = ABS(score2);
    } else {
        score2 = 0.;
    }

    if ((score0 == 0) && (score1 == 0) && (score2 == 0)) {
        matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
    } else if ((score0 > score1) && (score0 > score2)) {
        if ((c0 == 0) || (c0 == C0 - 1)) {
            matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
        } else {
            const float val = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
            const float val_left = mismatchedSdf[LOC(b, c0 - 1, c1, c2, C0, C1, C2)];
            const float val_right = mismatchedSdf[LOC(b, c0 + 1, c1, c2, C0, C1, C2)];
            if ((ABS(val_left) > ABS(val)) && (ABS(val) > ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft0 * ABS(val);
            } else if ((ABS(val_left) < ABS(val)) && (ABS(val) < ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft0 * ABS(val);
            } else {  // It will be determined by which side it is closer
                if ((ABS(val_left) - ABS(val) < ABS(val_right) - ABS(val))) {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft0 * ABS(val);
                } else {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft0 * ABS(val);
                }
            }
        }
    } else if ((score1 > score0) && (score1 > score2)) {
        if ((c1 == 0) || (c1 == C1 - 1)) {
            matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
        } else {
            const float val = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
            const float val_left = mismatchedSdf[LOC(b, c0, c1 - 1, c2, C0, C1, C2)];
            const float val_right = mismatchedSdf[LOC(b, c0, c1 + 1, c2, C0, C1, C2)];
            if ((ABS(val_left) > ABS(val)) && (ABS(val) > ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft1 * ABS(val);
            } else if ((ABS(val_left) < ABS(val)) && (ABS(val) < ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft1 * ABS(val);
            } else {  // It will be determined by which side it is closer
                if ((ABS(val_left) - ABS(val) < ABS(val_right) - ABS(val))) {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft1 * ABS(val);
                } else {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft1 * ABS(val);
                }
            }
        }
    } else if ((score2 > score0) && (score2 > score1)) {
        if ((c2 == 0) || (c2 == C2 - 1)) {
            matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
        } else {
            const float val = mismatchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)];
            const float val_left = mismatchedSdf[LOC(b, c0, c1, c2 - 1, C0, C1, C2)];
            const float val_right = mismatchedSdf[LOC(b, c0, c1, c2 + 1, C0, C1, C2)];
            if ((ABS(val_left) > ABS(val)) && (ABS(val) > ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft2 * ABS(val);
            } else if ((ABS(val_left) < ABS(val)) && (ABS(val) < ABS(val_right))) {
                matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft2 * ABS(val);
            } else {  // It will be determined by which side it is closer
                if ((ABS(val_left) - ABS(val) < ABS(val_right) - ABS(val))) {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = -signLeft2 * ABS(val);
                } else {
                    matchedSdf[LOC(b, c0, c1, c2, C0, C1, C2)] = signLeft2 * ABS(val);
                }
            }
        }
    }

}

}  // namespace


std::vector<at::Tensor> sdf_cuda(
        at::Tensor queryPoint,  // (b, q, 3)
        at::Tensor vert,  // (b, v, 3)
        at::Tensor face,  // (b, f, 3)
        at::Tensor referencePoints,  // (b, r, 3)
        at::Tensor sdf,   // (b, q)
        at::Tensor projPoint,  // (b, q, 3)
        at::Tensor cudaIntBuffer  // (b, q, dint)
    ) {

    const int b = queryPoint.size(0);
    const int q = queryPoint.size(1);
    const int nVert = vert.size(1);
    const int nFace = face.size(1);
    const int r = referencePoints.size(1);
    const int dint = cudaIntBuffer.size(2);
    CHECK_INPUT(queryPoint);
    CHECK_INPUT(vert);
    CHECK_INPUT(face);
    CHECK_INPUT(sdf);
    CHECK_INPUT(projPoint);
    CHECK_INPUT(referencePoints);
    CHECK_INPUT(cudaIntBuffer);
    TORCH_CHECK((queryPoint.dim() == 3) && (queryPoint.size(0) == b) &&
                (queryPoint.size(1) == q) && (queryPoint.size(2) == 3));
    TORCH_CHECK((vert.dim() == 3) && (vert.size(0) == b) && (vert.size(1) == nVert) && (vert.size(2) == 3));
    TORCH_CHECK((face.dim() == 3) && (face.size(0) == b) && (face.size(1) == nFace) && (face.size(2) == 3));
    TORCH_CHECK((sdf.dim() == 2) && (sdf.size(0) == b) && (sdf.size(1) == q));
    TORCH_CHECK((projPoint.dim() == 3) && (projPoint.size(0) == b) &&
                (projPoint.size(1) == q) && (projPoint.size(2) == 3));
    TORCH_CHECK((referencePoints.dim() == 3) && (referencePoints.size(0) == b) &&
                (referencePoints.size(1) == r) && (referencePoints.size(2) == 3));
    TORCH_CHECK((cudaIntBuffer.dim() == 3) && (cudaIntBuffer.size(0) == b) &&
                (cudaIntBuffer.size(1) == q) && (cudaIntBuffer.size(2) == dint));
    TORCH_CHECK(dint >= r);  // We need those buffer to store num_intersect.
    TORCH_CHECK(r % 2 == 1);  // So that we can break the tie.
    const int threads = 512;
    const dim3 blocks ((b * q - 1) / threads + 1);
    cudaSetDevice(queryPoint.get_device());

    AT_DISPATCH_FLOATING_TYPES(queryPoint.type(), "sdf_cuda", ([&] {
      sdf_cuda_kernel<scalar_t><<<blocks, threads>>>(
          queryPoint.data<scalar_t>(),
          vert.data<scalar_t>(),
          face.data<int32_t>(),
          referencePoints.data<scalar_t>(),
          sdf.data<scalar_t>(),
          projPoint.data<scalar_t>(),
          cudaIntBuffer.data<int32_t>(),
          b,
          q,
          nVert,
          nFace,
          r,
          dint);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sdf: %s\n", cudaGetErrorString(err));
    return {sdf, projPoint, cudaIntBuffer};
}


std::vector<at::Tensor> udf_cuda(
        at::Tensor queryPoint,  // (b, q, 3)
        at::Tensor vert,  // (b, v, 3)
        at::Tensor face,  // (b, f, 3)
        at::Tensor udf,   // (b, q)
        at::Tensor projPoint) {  // (b, q, 3)

    const int b = queryPoint.size(0);
    const int q = queryPoint.size(1);
    const int nVert = vert.size(1);
    const int nFace = face.size(1);
    CHECK_INPUT(queryPoint);
    CHECK_INPUT(vert);
    CHECK_INPUT(face);
    CHECK_INPUT(udf);
    CHECK_INPUT(projPoint);

    TORCH_CHECK((queryPoint.dim() == 3) && (queryPoint.size(0) == b) &&
                (queryPoint.size(1) == q) && (queryPoint.size(2) == 3));
    TORCH_CHECK((vert.dim() == 3) && (vert.size(0) == b) && (vert.size(1) == nVert)
                && (vert.size(2) == 3));
    TORCH_CHECK((face.dim() == 3) && (face.size(0) == b) && (face.size(1) == nFace)
                && (face.size(2) == 3));
    TORCH_CHECK((udf.dim() == 2) && (udf.size(0) == b) && (udf.size(1) == q));
    TORCH_CHECK((projPoint.dim() == 3) && (projPoint.size(0) == b) &&
                (projPoint.size(1) == q) && (projPoint.size(2) == 3));

    const int threads = 512;
    const dim3 blocks ((b * q - 1) / threads + 1);
    cudaSetDevice(queryPoint.get_device());

    AT_DISPATCH_FLOATING_TYPES(queryPoint.type(), "udf_cuda", ([&] {
      udf_cuda_kernel<scalar_t><<<blocks, threads>>>(
          queryPoint.data<scalar_t>(),
          vert.data<scalar_t>(),
          face.data<int32_t>(),
          udf.data<scalar_t>(),
          projPoint.data<scalar_t>(),
          b,
          q,
          nVert,
          nFace);
      }));

      cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sdf: %s\n", cudaGetErrorString(err));
    return {udf, projPoint};
}

std::vector<at::Tensor> sign_value_matching_cuda(
        at::Tensor mismatchedSdf,
        int radius,
        at::Tensor matchedSdf
) {
    assert(mismatchedSdf.dim() == 4);
    assert(matchedSdf.dim() == 4);
    const int B = mismatchedSdf.size(0);
    const int C0 = mismatchedSdf.size(1);
    const int C1 = mismatchedSdf.size(2);
    const int C2 = mismatchedSdf.size(3);
    assert(matchedSdf.size(0) == B);
    assert(matchedSdf.size(1) == C0);
    assert(matchedSdf.size(2) == C1);
    assert(matchedSdf.size(3) == C2);
    CHECK_INPUT(mismatchedSdf);
    CHECK_INPUT(matchedSdf);

    const int threads = 512;
    const dim3 blocks ((B * C0 * C1 * C2 - 1) / threads + 1);
    cudaSetDevice(mismatchedSdf.get_device());

    AT_DISPATCH_FLOATING_TYPES(mismatchedSdf.type(), "sign_value_matching_cuda", ([&] {
        sign_value_matching_cuda_kernel<scalar_t><<<blocks, threads>>>(
            mismatchedSdf.data<scalar_t>(),
            radius,
            B, C0, C1, C2,
            matchedSdf.data<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in sdf: %s\n", cudaGetErrorString(err));
    return {matchedSdf};
}
