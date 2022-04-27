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
__global__ void forward_face_index_map_cuda_kernel_1(
        const scalar_t* __restrict__ faces,
        scalar_t* __restrict__ faces_inv,
        int batch_size,
        int num_faces,
        int image_size,
        int capture_double_side) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t* face_inv_g = &faces_inv[i * 9];

    /* return if backside */
    if (capture_double_side == 0) {
        if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
            return;
    }

    /* p[num][xy]: x, y is normalized from [-1, 1] to [0, is - 1]. */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = 0.5 * (face[3 * num + dim] * is + is - 1);
        }
    }

    /* compute face_inv */
    scalar_t face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }
    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv_g[k] = face_inv[k];
    }
}

template <typename scalar_t>
__global__ void forward_face_index_map_cuda_kernel_2(
        const scalar_t* faces,
        const int32_t* face_flag,
        scalar_t* faces_inv,
        int32_t* __restrict__ face_index_map,
        scalar_t* __restrict__ weight_map,
        scalar_t* __restrict__ depth_map,
        scalar_t* __restrict__ face_inv_map,
        int batch_size,
        int num_faces,
        int image_size,
        scalar_t near,
        scalar_t far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = pn / is;
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1 - is) / is;
    const scalar_t xp = (2. * xi + 1 - is) / is;
    
    const scalar_t* face = &faces[bn * nf * 9] - 9;
    // const int32_t* face_flag_now = &face_flag[bn * nf * 9] - 1;  # BUG!!
    const int32_t* face_flag_now = &face_flag[bn * nf] - 1;
    scalar_t* face_inv = &faces_inv[bn * nf * 9] - 9;
    scalar_t depth_min = far;
    int face_index_min = -1;
    scalar_t weight_min[3];
    scalar_t face_inv_min[9];
    for (int fn = 0; fn < nf; fn++) {
        /* go to next face */
        face += 9;
        face_inv += 9;
        face_flag_now += 1;

        /* return if not valid indicated by the flag. */
        if (*face_flag_now == 0)
            continue;

        /* return if backside */
        if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
            continue;

        /* check [py, px] is inside the face */
        if (((yp - face[1]) * (face[3] - face[0]) < (xp - face[0]) * (face[4] - face[1])) ||
            ((yp - face[4]) * (face[6] - face[3]) < (xp - face[3]) * (face[7] - face[4])) ||
            ((yp - face[7]) * (face[0] - face[6]) < (xp - face[6]) * (face[1] - face[7])))
            continue;

        /* compute w = face_inv * p */
        scalar_t w[3];
        w[0] = face_inv[3 * 0 + 0] * xi + face_inv[3 * 0 + 1] * yi + face_inv[3 * 0 + 2];
        w[1] = face_inv[3 * 1 + 0] * xi + face_inv[3 * 1 + 1] * yi + face_inv[3 * 1 + 2];
        w[2] = face_inv[3 * 2 + 0] * xi + face_inv[3 * 2 + 1] * yi + face_inv[3 * 2 + 2];

        /* sum(w) -> 1, 0 < w < 1 */
        scalar_t w_sum = 0;
        for (int k = 0; k < 3; k++) {
            w[k] = min(max(w[k], 0.), 1.);
            w_sum += w[k];
        }
        for (int k = 0; k < 3; k++) {
            w[k] /= w_sum;
        }
        /* compute 1 / zp = sum(w / z) */
        const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
        if (zp <= near || far <= zp) {
            continue;
        }

        /* check z-buffer */
        if (zp < depth_min) {
            depth_min = zp;
            face_index_min = fn;
            for (int k = 0; k < 3; k++) {
                weight_min[k] = w[k];
            }
            if (return_depth) {
                for (int k = 0; k < 9; k++) {
                    face_inv_min[k] = face_inv[k];
                }
            }
        }
    }

    /* set to global memory */
    if ((0 <= face_index_min)) {
        depth_map[i] = depth_min;
        face_index_map[i] = face_index_min;
        for (int k = 0; k < 3; k++) {
            weight_map[3 * i + k] = weight_min[k];
        }
        if (return_depth) {
            for (int k = 0; k < 9; k++) {
                face_inv_map[9 * i + k] = face_inv_min[k];
            }
        }
    }
}

template <typename scalar_t>
__global__ void forward_texture_sampling_cuda_kernel(
		const scalar_t* faces,
		const scalar_t* textures,
		const int32_t* face_index_map,
		const scalar_t* weight_map,
		const scalar_t* depth_map,
		scalar_t* rgb_map,
		int32_t* sampling_index_map,
        scalar_t* sampling_weight_map,
        size_t batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        scalar_t eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
        */
        const int bn = i / (image_size * image_size);
        const int nf = num_faces;
        const int ts = texture_size;
        const scalar_t* face = &faces[face_index * 9];
        const scalar_t* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* pixel = &rgb_map[i * 3];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t depth = depth_map[i];
        int32_t* sampling_indices = &sampling_index_map[i * 8];
        scalar_t* sampling_weights = &sampling_weight_map[i * 8];
    
        /* get texture index (float) */
        scalar_t texture_index_float[3];
        for (int k = 0; k < 3; k++) { scalar_t tif = weight[k] * (ts - 1) * (depth / (face[3 * k + 2]));
            tif = max(tif, 0.);
            tif = min(tif, ts - 1 - eps);
            texture_index_float[k] = tif;
        }
    
        /* blend */
        scalar_t new_pixel[3] = {0, 0, 0};
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = 1;                         // weight
            int texture_index_int[3];            // index in source (int)
            for (int k = 0; k < 3; k++) {
                if ((pn >> k) % 2 == 0) {
                    w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                    texture_index_int[k] = (int)texture_index_float[k];
                }
                else {
                    w *= texture_index_float[k] - (int)texture_index_float[k];
                    texture_index_int[k] = (int)texture_index_float[k] + 1;
                }
            }
    
            int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
            for (int k = 0; k < 3; k++)
                new_pixel[k] += w * texture[isc * 3 + k];
            sampling_indices[pn] = isc;
            sampling_weights[pn] = w;
        }
        for (int k = 0; k < 3; k++)
            pixel[k] = new_pixel[k];
    }
}

template <typename scalar_t>
__global__ void backward_pixel_map_cuda_kernel(
		const scalar_t* faces,
        int32_t*  face_index_map,
        scalar_t*  rgb_map,
        scalar_t*  alpha_map,
        scalar_t*  grad_rgb_map,
        scalar_t*  grad_alpha_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        scalar_t eps,
        int return_rgb,
        int return_alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int bn = i / num_faces;
    const int fn = i % num_faces;
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t grad_face[9] = {};

    /* check backside */
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        return;

    /* for each edge */
    for (int edge_num = 0; edge_num < 3; edge_num++) {
        /* set points of target edge */
        int pi[3];
        scalar_t pp[3][2];
        for (int num = 0; num < 3; num++)
            pi[num] = (edge_num + num) % 3;
        for (int num = 0; num < 3; num++) {
            for (int dim = 0; dim < 2; dim++) {
                pp[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
            }
        }

        /* for dy, dx */
        for (int axis = 0; axis < 2; axis++) {
            /* */
            scalar_t p[3][2];
            for (int num = 0; num < 3; num++) {
                for (int dim = 0; dim < 2; dim++) {
                    p[num][dim] = pp[num][(dim + axis) % 2];
                }
            }

            /* set direction */
            int direction;
            if (axis == 0) {
                if (p[0][0] < p[1][0])
                    direction = -1;
                else
                    direction = 1;
            } else {
                if (p[0][0] < p[1][0])
                    direction = 1;
                else
                    direction = -1;
            }

            /* along edge */
            int d0_from, d0_to;
            d0_from = max(ceil(min(p[0][0], p[1][0])), 0.);
            d0_to = min(max(p[0][0], p[1][0]), is - 1.);
            for (int d0 = d0_from; d0 <= d0_to; d0++) {
                /* get cross point */
                int d1_in, d1_out;
                const scalar_t d1_cross = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                if (0 < direction)
                    d1_in = floor(d1_cross);
                else
                    d1_in = ceil(d1_cross);
                d1_out = d1_in + direction;

                /* continue if cross point is not shown */
                if (d1_in < 0 || is <= d1_in)
                    continue;
                if (d1_out < 0 || is <= d1_out)
                    continue;

                /* get color of in-pixel and out-pixel */
                scalar_t alpha_in;
                scalar_t alpha_out;
                scalar_t *rgb_in;
                scalar_t *rgb_out;
                int map_index_in, map_index_out;
                if (axis == 0) {
                    map_index_in = bn * is * is + d1_in * is + d0;
                    map_index_out = bn * is * is + d1_out * is + d0;
                }
                else {
                    map_index_in = bn * is * is + d0 * is + d1_in;
                    map_index_out = bn * is * is + d0 * is + d1_out;
                }
                if (return_alpha) {
                    alpha_in = alpha_map[map_index_in];
                    alpha_out = alpha_map[map_index_out];
                }
                if (return_rgb) {
                    rgb_in = &rgb_map[map_index_in * 3];
                    rgb_out = &rgb_map[map_index_out * 3];
                }

                /* out */
                bool is_in_fn = (face_index_map[map_index_in] == fn);
                if (is_in_fn) {
                    int d1_limit;
                    if (0 < direction)
                        d1_limit = is - 1;
                    else
                        d1_limit = 0;
                    int d1_from = max(min(d1_out, d1_limit), 0);
                    int d1_to = min(max(d1_out, d1_limit), is - 1);
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_offset, map_index_from;
                    if (axis == 0) {
                        map_offset = is;
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_offset = 1;
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from];
                        grad_alpha_map_p = &grad_alpha_map[map_index_from];
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3];
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3];
                    }
                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        scalar_t diff_grad = 0;
                        if (return_alpha) {
                            diff_grad += (*alpha_map_p - alpha_in) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                                diff_grad += (rgb_map_p[k] - rgb_in[k]) * grad_rgb_map_p[k];
                        }
                        if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        if (diff_grad <= 0)
                            continue;
                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }

                /* in */
                {
                    int d1_limit;
                    scalar_t d0_cross2;
                    if ((d0 - p[0][0]) * (d0 - p[2][0]) < 0) {
                        d0_cross2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                    }
                    else {
                        d0_cross2 = (p[1][1] - p[2][1]) / (p[1][0] - p[2][0]) * (d0 - p[2][0]) + p[2][1];
                    }
                    if (0 < direction)
                        d1_limit = ceil(d0_cross2);
                    else
                        d1_limit = floor(d0_cross2);
                    int d1_from = max(min(d1_in, d1_limit), 0);
                    int d1_to = min(max(d1_in, d1_limit), is - 1);

                    int* face_index_map_p;
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_index_from;
                    int map_offset;
                    if (axis == 0)
                        map_offset = is;
                    else
                        map_offset = 1;
                    if (axis == 0) {
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    face_index_map_p = &face_index_map[map_index_from] - map_offset;
                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from] - map_offset;
                        grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3] - 3 * map_offset;
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3] - 3 * map_offset;
                    }

                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        face_index_map_p += map_offset;
                        if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        if (*face_index_map_p != fn)
                            continue;

                        scalar_t diff_grad = 0;
                        if (return_alpha) {
                            diff_grad += (*alpha_map_p - alpha_out) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                                diff_grad += (rgb_map_p[k] - rgb_out[k]) * grad_rgb_map_p[k];
                        }
                        if (diff_grad <= 0)
                            continue;

                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }
            }
        }
    }

    /* set to global gradient variable */
    for (int k = 0; k < 9; k++)
        grad_faces[i * 9 + k] = grad_face[k];
}

template <typename scalar_t>
__global__ void backward_textures_cuda_kernel(
        const int32_t* face_index_map,
        scalar_t* sampling_weight_map,
        int32_t* sampling_index_map,
        scalar_t* grad_rgb_map,
        scalar_t* grad_textures,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        size_t texture_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    if (0 <= face_index) {
        int is = image_size;
        int nf = num_faces;
        int ts = texture_size;
        int bn = i / (is * is);    // batch number [0 -> bs]
    
        scalar_t* grad_texture = &grad_textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
        int* sampling_index_map_p = &sampling_index_map[i * 8];
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = *sampling_weight_map_p++;
            int isc = *sampling_index_map_p++;
            scalar_t* grad_texture_p = &grad_texture[isc * 3];
            scalar_t* grad_rgb_map_p = &grad_rgb_map[i * 3];
            for (int k = 0; k < 3; k++)
                atomicAdd(grad_texture_p++, w * *grad_rgb_map_p++);
        }
    }
}

template <typename scalar_t>
__global__ void backward_depth_map_cuda_kernel(
        const scalar_t*  faces,
        const scalar_t*  depth_map,
        const int32_t* face_index_map,
        const scalar_t* face_inv_map,
        const scalar_t* weight_map,
        scalar_t*  grad_depth_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int fn = face_index_map[i];
    if (0 <= fn) {
        const int nf = num_faces;
        const int is = image_size;
        const int bn = i / (is * is);
        const scalar_t* face = &faces[(bn * nf + fn) * 9];
        const scalar_t depth = depth_map[i];
        const scalar_t depth2 = depth * depth;
        const scalar_t* face_inv = &face_inv_map[i * 9];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t grad_depth = grad_depth_map[i];
        scalar_t* grad_face = &grad_faces[(bn * nf + fn) * 9];
    
        /* derivative wrt z */
        for (int k = 0; k < 3; k++) {
            const scalar_t z_k = face[3 * k + 2];
            atomicAdd(&grad_face[3 * k + 2], grad_depth * weight[k] * depth2 / (z_k * z_k));
        }
    
        /* derivative wrt x, y */
        scalar_t tmp[3] = {};
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                tmp[k] += -face_inv[3 * l + k] / face[3 * l + 2];
            }
        }
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
            // k: point number, l: dimension
            atomicAdd(&grad_face[3 * k + l], -grad_depth * tmp[l] * weight[k] * depth2 * is / 2);
            }
        }
    }
}

// zhuzhu
template <typename scalar_t>
__global__ void forward_face_info_onto_map_cuda_kernel(
        const int32_t* face_index_map,
        const scalar_t* face_info,
        const scalar_t* background,
        scalar_t* __restrict__ map_info,
        int image_height,
        int image_width,
        int n_faces,
        int dim_info,
        int batch_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }
    int sample_insideBatch_id = i / (image_height * image_width);

    int face_id = face_index_map[i];
    if ((face_id >= 0) && (face_id < n_faces)) {
        for (int j = 0; j < dim_info; j++) {
            map_info[i * dim_info + j] =
                face_info[sample_insideBatch_id * n_faces * dim_info + face_id * dim_info + j];
        }
    } else {
        for (int j = 0; j < dim_info; j++) {
            map_info[i * dim_info + j] = background[j];
        }
    }
}

// zhuzhu
template <typename scalar_t>
__global__ void forward_vert_info_onto_map_cuda_kernel(
        const int32_t* face_index_map,
        const scalar_t* weight_map,
        const int32_t* tris,
        const scalar_t* vert_info,
        const scalar_t* background,
        scalar_t* __restrict__ map_info,
        int image_height,
        int image_width,
        int n_faces,
        int n_verts,
        int dim_info,
        int batch_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }
    int face_id = face_index_map[i];
    int sample_insideBatch_id = i / (image_height * image_width);
    if ((face_id >= 0) && (face_id < n_faces)) {
        int vert0_id = tris[sample_insideBatch_id * n_faces * 3 + face_id * 3 + 0];
        int vert1_id = tris[sample_insideBatch_id * n_faces * 3 + face_id * 3 + 1];
        int vert2_id = tris[sample_insideBatch_id * n_faces * 3 + face_id * 3 + 2];
        scalar_t bary0 = weight_map[i * 3 + 0];
        scalar_t bary1 = weight_map[i * 3 + 1];
        scalar_t bary2 = weight_map[i * 3 + 2];
        for (int j = 0; j < dim_info; j++) {
            map_info[i * dim_info + j] =
                bary0 * vert_info[sample_insideBatch_id * n_verts * dim_info + vert0_id * dim_info + j] +
                bary1 * vert_info[sample_insideBatch_id * n_verts * dim_info + vert1_id * dim_info + j] +
                bary2 * vert_info[sample_insideBatch_id * n_verts * dim_info + vert2_id * dim_info + j];
        }
    } else {
        for (int j = 0; j < dim_info; j++) {
            map_info[i * dim_info + j] = background[j];
        }
    }
}

// zhuzhu
#define LOC(sample_insideBatch_id, y, x, image_height, image_width) ((sample_insideBatch_id) * (image_height) * (image_width) + (y) * (image_width) + (x))
template <typename scalar_t>
__global__ void forward_grid_sampling_cuda_kernel(
        const scalar_t* infoA,
        const scalar_t* mapB_of_xyA,
        const int32_t* maskB,
        const scalar_t* background,
        scalar_t* __restrict__ infoB,
        bool pixel_is_area,
        int interp_method_id, // 0 leftUp, 1 rightUp, 2 leftBottom, 3 rightBottom, 4 weightedAverage
        int image_height_A,
        int image_width_A,
        int image_height_B,
        int image_width_B,
        int dim_info,
        int batch_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height_B * image_width_B) {
        return;
    }
    if (maskB[i] == 0) {
        for (int j = 0; j < dim_info; j++) {
            infoB[i * dim_info + j] = background[j];
        }
        return;
    }
    int sample_insideBatch_id = i / (image_height_B * image_width_B);

    float x_in_A = mapB_of_xyA[i * 2 + 0];
    float y_in_A = mapB_of_xyA[i * 2 + 1];
    if (!pixel_is_area) {
        x_in_A -= 1. / image_width_A;
        y_in_A -= 1. / image_height_A;
    }
    assert((x_in_A > -1.) && (x_in_A < 1. - 2. / image_width_A) && (y_in_A > -1.) && (y_in_A < 1. - 2. / image_height_A));
    float xCrop_in_A = (x_in_A + 1.) * image_width_A / 2.;
    float yCrop_in_A = (y_in_A + 1.) * image_height_A / 2.;
    int x_left = (int)xCrop_in_A;
    int x_right = x_left + 1;
    int y_up = (int)yCrop_in_A;
    int y_bottom = y_up + 1;
    for (int j = 0; j < dim_info; j++) {
        scalar_t infoA_leftUp = infoA[LOC(sample_insideBatch_id, y_up, x_left, image_height_A, image_width_A) * dim_info + j];
        scalar_t infoA_rightUp = infoA[LOC(sample_insideBatch_id, y_up, x_right, image_height_A, image_width_A) * dim_info + j];
        scalar_t infoA_leftBottom = infoA[LOC(sample_insideBatch_id, y_bottom, x_left, image_height_A, image_width_A) * dim_info + j];
        scalar_t infoA_rightBottom = infoA[LOC(sample_insideBatch_id, y_bottom, x_right, image_height_A, image_width_A) * dim_info + j];
        if (interp_method_id == 0) {
            infoB[i * dim_info + j] = infoA_leftUp;
        } else if (interp_method_id == 1) {
            infoB[i * dim_info + j] = infoA_rightUp;
        } else if (interp_method_id == 2) {
            infoB[i * dim_info + j] = infoA_leftBottom;
        } else if (interp_method_id == 3) {
            infoB[i * dim_info + j] = infoA_rightBottom;
        } else if (interp_method_id == 4) {
            scalar_t infoA_left = (y_bottom - yCrop_in_A) * infoA_leftUp + (yCrop_in_A - y_up) * infoA_leftBottom;
            scalar_t infoA_right = (y_bottom - yCrop_in_A) * infoA_rightUp + (yCrop_in_A - y_up) * infoA_rightBottom;
            infoB[i * dim_info + j] = (x_right - xCrop_in_A) * infoA_left + (xCrop_in_A - x_left) * infoA_right;
        } else {
            assert(1 + 1 != 2);  // not implemented error
        }
    }
}

// zhuzhu
#define LOC_FACES_XY(sample_insideBatch_id, face_id, v012, xy, n_faces) ((sample_insideBatch_id) * (n_faces) * 6 + (face_id) * 6 + (v012) * 2 + (xy))
template <typename scalar_t>
__global__ void forward_get_edge_mask_cuda_kernel(
        const scalar_t* faces_xy,  // m * n_faces * 3 * 2
        const int32_t* face_index_map,  // m * image_height * image_width
        scalar_t* __restrict__ edge_mask,  // m * image_height * image_width
        bool pixel_is_area,
        bool soft_mode,
        scalar_t line_width_radius,
        int image_height,
        int image_width,
        int n_faces,
        int batch_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }
    const int sample_insideBatch_id = i / (image_height * image_width);
    const int face_id = face_index_map[i];
    const int this_col = i % image_width;
    const int this_row = (i / image_width) % image_height;
    scalar_t this_x, this_y;
    if (pixel_is_area) {
        this_x = (float)this_col / image_width * 2. - 1.;
        this_y = (float)this_row / image_height * 2. - 1.;
    } else {  // False, so that the pixel is represented by the center
        this_x = (this_col + .5) / image_width * 2. - 1.;
        this_y = (this_row + .5) / image_height * 2. - 1.;
    }
    if ((face_id >= 0) && (face_id < n_faces)) {
        scalar_t flag_max = 0.;
        for (int j = 0; j < 3; j++) {  // three edges of one triangle
            const int p = (j + 1) % 3;
            const int q = (j + 2) % 3;
            const scalar_t a_x = faces_xy[LOC_FACES_XY(sample_insideBatch_id, face_id, p, 0, n_faces)];
            const scalar_t a_y = faces_xy[LOC_FACES_XY(sample_insideBatch_id, face_id, p, 1, n_faces)];
            const scalar_t b_x = faces_xy[LOC_FACES_XY(sample_insideBatch_id, face_id, q, 0, n_faces)];
            const scalar_t b_y = faces_xy[LOC_FACES_XY(sample_insideBatch_id, face_id, q, 1, n_faces)];
            const scalar_t lambda = ((this_x - a_x) * (b_x - a_x) + (this_y - a_y) * (b_y - a_y)) /
                ((b_x - a_x) * (b_x - a_x) + (b_y - a_y) * (b_y - a_y));
            if ((lambda < 0.) || (lambda > 1.)) {
                continue;
            }
            const scalar_t t_x = (1 - lambda) * a_x + lambda * b_x;
            const scalar_t t_y = (1 - lambda) * a_y + lambda * b_y;
            const scalar_t d2 = (this_x - t_x) * (this_x - t_x) + (this_y - t_y) * (this_y - t_y);
            scalar_t flag = 0.;
            if (!soft_mode) {
                if (d2 < line_width_radius * line_width_radius) {
                    flag = 1.;
                }
            } else {
                flag = exp(-.5 * d2 / (line_width_radius * line_width_radius));
            }
            if (flag > flag_max) {
                flag_max = flag;
            }
        }
        edge_mask[i] = flag_max;
    } else {
        edge_mask[i] = 0;
    }
}
}

// zhuzhu
at::Tensor forward_face_info_onto_map_cuda(at::Tensor face_index_map,
                                           at::Tensor face_info,
                                           at::Tensor background,
                                           at::Tensor map_info) {
    const auto image_height = face_index_map.size(1);
    const auto image_width = face_index_map.size(2);
    const auto n_faces = face_info.size(1);
    const auto dim_info = face_info.size(2);
    const auto batch_size = face_index_map.size(0);

    assert(face_info.size(0) == batch_size);
    assert(background.size(0) == dim_info);
    assert((map_info.size(0) == batch_size) && (map_info.size(1) == image_height) && (map_info.size(2) == image_width)
        && (map_info.size(3) == dim_info));

    const int threads = 512;
    const dim3 blocks ((batch_size * image_height * image_width - 1) / threads + 1);

    cudaSetDevice(face_index_map.get_device());

    AT_DISPATCH_FLOATING_TYPES(face_info.type(), "forward_face_info_onto_map_cuda", ([&] {
        forward_face_info_onto_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
            face_index_map.data<int32_t>(),
            face_info.data<scalar_t>(),
            background.data<scalar_t>(),
            map_info.data<scalar_t>(),
            image_height,
            image_width,
            n_faces,
            dim_info,
            batch_size
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_face_info_onto_map: %s\n", cudaGetErrorString(err));

    return map_info;
}

// zhuzhu
at::Tensor forward_vert_info_onto_map_cuda(at::Tensor face_index_map,
                                           at::Tensor weight_map,
                                           at::Tensor tris,
                                           at::Tensor vert_info,
                                           at::Tensor background,
                                           at::Tensor map_info) {

    const auto image_height = face_index_map.size(1);
    const auto image_width = face_index_map.size(2);
    const auto n_faces = tris.size(1);
    const auto n_verts = vert_info.size(1);
    const auto dim_info = vert_info.size(2);
    const auto batch_size = face_index_map.size(0);

    assert((weight_map.size(0) == batch_size) && (weight_map.size(1) == image_height)
        && (weight_map.size(2) == image_width) && (weight_map.size(3) == 3));
    assert((tris.size(0) == batch_size) && (tris.size(2) == 3));
    assert(vert_info.size(0) == batch_size);
    assert(background.size(0) == dim_info);
    assert((map_info.size(0) == batch_size) && (map_info.size(1) == image_height)
        && (map_info.size(2) == image_width) && (map_info.size(3) == dim_info));

    const int threads = 512;
    const dim3 blocks((batch_size * image_height * image_width - 1) / threads + 1);

    cudaSetDevice(face_index_map.get_device());

    AT_DISPATCH_FLOATING_TYPES(vert_info.type(), "forward_vert_info_onto_map_cuda", ([&] {
        forward_vert_info_onto_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
            face_index_map.data<int32_t>(),
            weight_map.data<scalar_t>(),
            tris.data<int32_t>(),
            vert_info.data<scalar_t>(),
            background.data<scalar_t>(),
            map_info.data<scalar_t>(),
            image_height,
            image_width,
            n_faces,
            n_verts,
            dim_info,
            batch_size
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_face_info_onto_map: %s\n", cudaGetErrorString(err));

    return map_info;
}

// zhuzhu
at::Tensor forward_grid_sampling_cuda(
        at::Tensor infoA,
        at::Tensor mapB_of_xyA,
        at::Tensor maskB,
        at::Tensor background,
        at::Tensor infoB,
        bool pixel_is_area,
        int interp_method_id) {

    const auto image_height_A = infoA.size(1);
    const auto image_width_A = infoA.size(2);
    const auto image_height_B = mapB_of_xyA.size(1);
    const auto image_width_B = mapB_of_xyA.size(2);
    const auto dim_info = infoA.size(3);
    const auto batch_size = infoA.size(0);

    assert((mapB_of_xyA.size(0) == batch_size) && (mapB_of_xyA.size(3) == 2));
    assert((maskB.size(0) == batch_size) && (maskB.size(1) == image_height_B) &&
        (maskB.size(2) == image_width_B));
    assert(background.size(0) == dim_info);
    assert((infoB.size(0) == batch_size) && (infoB.size(1) == image_height_B) &&
        (infoB.size(2) == image_width_B) && (infoB.size(3) == dim_info));

    const int threads = 512;
    const dim3 blocks((batch_size * image_height_B * image_width_B - 1) / threads + 1);

    cudaSetDevice(infoA.get_device());

    AT_DISPATCH_FLOATING_TYPES(infoA.type(), "forward_grid_sampling_cuda", ([&] {
        forward_grid_sampling_cuda_kernel<scalar_t><<<blocks, threads>>>(
            infoA.data<scalar_t>(),
            mapB_of_xyA.data<scalar_t>(),
            maskB.data<int32_t>(),
            background.data<scalar_t>(),
            infoB.data<scalar_t>(),
            pixel_is_area,
            interp_method_id,
            image_height_A,
            image_width_A,
            image_height_B,
            image_width_B,
            dim_info,
            batch_size
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_face_info_onto_map: %s\n", cudaGetErrorString(err));

    return infoB;
}

// zhuzhu
at::Tensor forward_get_edge_mask_cuda(
        at::Tensor faces_xy,
        at::Tensor face_index_map,
        at::Tensor edge_mask,
        bool pixel_is_area,
        bool soft_mode,
        float line_width_radius) {

    const auto batch_size = faces_xy.size(0);
    const auto n_faces = faces_xy.size(1);
    const auto image_height = face_index_map.size(1);
    const auto image_width = face_index_map.size(2);

    assert((faces_xy.size(2) == 3) && (faces_xy.size(3) == 2));
    assert(face_index_map.size(0) == batch_size);
    assert((edge_mask.size(0) == batch_size) && (edge_mask.size(1) == image_height) && (edge_mask.size(2) == image_width));
    assert(line_width_radius >= 0);

    const int threads = 512;
    const dim3 blocks((batch_size * image_height * image_width - 1) / threads + 1);

    cudaSetDevice(faces_xy.get_device());

    AT_DISPATCH_FLOATING_TYPES(faces_xy.type(), "forward_get_edge_mask_cuda", ([&] {
        forward_get_edge_mask_cuda_kernel<scalar_t><<<blocks, threads>>>(
            faces_xy.data<scalar_t>(),
            face_index_map.data<int32_t>(),
            edge_mask.data<scalar_t>(),
            pixel_is_area,
            soft_mode,
            (scalar_t)line_width_radius,
            image_height,
            image_width,
            n_faces,
            batch_size
        );
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_face_info_onto_map: %s\n", cudaGetErrorString(err));

    return edge_mask;
}

std::vector<at::Tensor> forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_flag,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor face_inv_map,
        at::Tensor faces_inv,
        int image_size,
        float near,
        float far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);

    cudaSetDevice(faces.get_device());

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda_1", ([&] {
      forward_face_index_map_cuda_kernel_1<scalar_t><<<blocks_1, threads>>>(
          faces.data<scalar_t>(),
          faces_inv.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          0);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map_1: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((batch_size * image_size * image_size - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda_2", ([&] {
      forward_face_index_map_cuda_kernel_2<scalar_t><<<blocks_2, threads>>>(
          faces.data<scalar_t>(),
          face_flag.data<int32_t>(),
          faces_inv.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          face_inv_map.data<scalar_t>(),
          (int) batch_size,
          (int) num_faces,
          (int) image_size,
          (scalar_t) near,
          (scalar_t) far,
          return_rgb,
          return_alpha,
          return_depth);
      }));

    err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in forward_face_index_map_2: %s\n", cudaGetErrorString(err));
    return {face_index_map, weight_map, depth_map, face_inv_map};
}

std::vector<at::Tensor> forward_texture_sampling_cuda( at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    cudaSetDevice(textures.get_device());

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_texture_sampling_cuda", ([&] {
      forward_texture_sampling_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          rgb_map.data<scalar_t>(),
		  sampling_index_map.data<int32_t>(),
		  sampling_weight_map.data<scalar_t>(),
          batch_size,
		  num_faces,
          image_size,
          texture_size,
          eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_texture_sampling: %s\n", cudaGetErrorString(err));

    return {rgb_map, sampling_index_map, sampling_weight_map};
}

at::Tensor backward_pixel_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor rgb_map,
        at::Tensor alpha_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps,
        int return_rgb,
        int return_alpha) {
    
    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * num_faces - 1) / threads + 1);

    cudaSetDevice(faces.get_device());

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_pixel_map_cuda", ([&] {
      backward_pixel_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          rgb_map.data<scalar_t>(),
          alpha_map.data<scalar_t>(),
          grad_rgb_map.data<scalar_t>(),
          grad_alpha_map.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          batch_size,
		  num_faces,
          image_size,
          (scalar_t) eps,
          return_rgb,
          return_alpha);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_pixel_map: %s\n", cudaGetErrorString(err));

    return grad_faces;
}

at::Tensor backward_textures_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces) {

    const auto batch_size = face_index_map.size(0);
    const auto image_size = face_index_map.size(1);
    const auto texture_size = grad_textures.size(2);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    cudaSetDevice(face_index_map.get_device());

    AT_DISPATCH_FLOATING_TYPES(sampling_weight_map.type(), "backward_textures_cuda", ([&] {
      backward_textures_cuda_kernel<scalar_t><<<blocks, threads>>>(
          face_index_map.data<int32_t>(),
          sampling_weight_map.data<scalar_t>(),
          sampling_index_map.data<int32_t>(),
          grad_rgb_map.data<scalar_t>(),
          grad_textures.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_textures: %s\n", cudaGetErrorString(err));

    return grad_textures;
}
at::Tensor backward_depth_map_cuda(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    cudaSetDevice(faces.get_device());

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_depth_map_cuda", ([&] {
      backward_depth_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          face_inv_map.data<scalar_t>(),
          weight_map.data<scalar_t>(),
          grad_depth_map.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_depth_map: %s\n", cudaGetErrorString(err));

    return grad_faces;
}
