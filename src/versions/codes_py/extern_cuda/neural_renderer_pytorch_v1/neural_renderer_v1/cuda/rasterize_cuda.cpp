#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

// zhuzhu
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
        int return_depth);

std::vector<at::Tensor> forward_texture_sampling_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps);

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
        int return_alpha);

at::Tensor backward_textures_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces);

at::Tensor backward_depth_map_cuda(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// zhuzhu
at::Tensor forward_face_info_onto_map_cuda(at::Tensor face_index_map,
                                           at::Tensor face_info,
                                           at::Tensor background,
                                           at::Tensor map_info);

// zhuzhu
at::Tensor forward_vert_info_onto_map_cuda(at::Tensor face_index_map,
                                           at::Tensor weight_map,
                                           at::Tensor tris,
                                           at::Tensor vert_info,
                                           at::Tensor background,
                                           at::Tensor map_info);

// zhuzhu
at::Tensor forward_grid_sampling_cuda(at::Tensor infoA,
                                      at::Tensor mapB_of_xyA,
                                      at::Tensor maskB,
                                      at::Tensor background,
                                      at::Tensor infoB,
                                      bool pixel_is_area,
                                      int interp_method_id);

// zhuzhu
at::Tensor forward_get_edge_mask_cuda(at::Tensor faces_xy,
                                      at::Tensor face_index_map,
                                      at::Tensor mask_edge,
                                      bool pixel_is_area,
                                      bool soft_mode,
                                      float line_width_radius);

// zhuzhu
at::Tensor forward_face_info_onto_map(
        at::Tensor face_index_map,
        at::Tensor face_info,
        at::Tensor background,
        at::Tensor map_info) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(face_info);
    CHECK_INPUT(background);
    CHECK_INPUT(map_info);

    return forward_face_info_onto_map_cuda(face_index_map,
                                           face_info,
                                           background,
                                           map_info);
}

// zhuzhu
at::Tensor forward_vert_info_onto_map(
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor tris,
        at::Tensor vert_info,
        at::Tensor background,
        at::Tensor map_info) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(tris);
    CHECK_INPUT(vert_info);
    CHECK_INPUT(background);
    CHECK_INPUT(map_info);

    return forward_vert_info_onto_map_cuda(face_index_map,
                                           weight_map,
                                           tris,
                                           vert_info,
                                           background,
                                           map_info);
}

// zhuzhu
at::Tensor forward_grid_sampling(
        at::Tensor infoA,
        at::Tensor mapB_of_xyA,
        at::Tensor maskB,  // whether or not to grid-sample
        at::Tensor background,
        at::Tensor infoB,  // the output
        bool pixel_is_area,
        int interp_method_id) {

    CHECK_INPUT(infoA);
    CHECK_INPUT(mapB_of_xyA);
    CHECK_INPUT(maskB);
    CHECK_INPUT(background);
    CHECK_INPUT(infoB);

    return forward_grid_sampling_cuda(infoA,
                                      mapB_of_xyA,
                                      maskB,
                                      background,
                                      infoB,
                                      pixel_is_area,
                                      interp_method_id);
}

// zhuzhu
at::Tensor forward_get_edge_mask(
        at::Tensor faces_xy,  // m * n_faces * 3 * 2
        at::Tensor face_index_map,
        at::Tensor mask_edge,
        bool pixel_is_area,
        bool soft_mode,
        float line_width_radius) {

    CHECK_INPUT(faces_xy);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(mask_edge);

    return forward_get_edge_mask_cuda(faces_xy,
                                      face_index_map,
                                      mask_edge,
                                      pixel_is_area,
                                      soft_mode,
                                      line_width_radius);
}

// zhuzhu
std::vector<at::Tensor> forward_face_index_map(
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

    CHECK_INPUT(faces);
    CHECK_INPUT(face_flag);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(faces_inv);

    return forward_face_index_map_cuda(faces, face_flag, face_index_map, weight_map,
                                       depth_map, face_inv_map, faces_inv,
                                       image_size, near, far,
                                       return_rgb, return_alpha, return_depth);
}

std::vector<at::Tensor> forward_texture_sampling(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(sampling_weight_map);

    return forward_texture_sampling_cuda(faces, textures, face_index_map,
                                    weight_map, depth_map, rgb_map,
                                    sampling_index_map, sampling_weight_map,
                                    image_size, eps);
}

at::Tensor backward_pixel_map(
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

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(alpha_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_alpha_map);
    CHECK_INPUT(grad_faces);

    return backward_pixel_map_cuda(faces, face_index_map, rgb_map, alpha_map,
                                   grad_rgb_map, grad_alpha_map, grad_faces,
                                   image_size, eps, return_rgb, return_alpha);
}

at::Tensor backward_textures(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(sampling_weight_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_textures);

    return backward_textures_cuda(face_index_map, sampling_weight_map,
                                  sampling_index_map, grad_rgb_map,
                                  grad_textures, num_faces);
}

at::Tensor backward_depth_map(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(grad_depth_map);
    CHECK_INPUT(grad_faces);

    return backward_depth_map_cuda(faces, depth_map, face_index_map,
                                   face_inv_map, weight_map,
                                   grad_depth_map, grad_faces,
                                   image_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_face_info_onto_map", &forward_face_info_onto_map, "FORWARD_FACE_INFO_ONTO_MAP (CUDA)"); // zhuzhu
    m.def("forward_vert_info_onto_map", &forward_vert_info_onto_map, "FORWARD_VERT_INFO_ONTO_MAP (CUDA)"); // zhuzhu
    m.def("forward_grid_sampling", &forward_grid_sampling, "FORWARD_PIXEL_INTERP_GRABBING (CUDA)");  // zhuzhu
    m.def("forward_get_edge_mask", &forward_get_edge_mask, "FORWARD_GET_EDGE_MASK (CUDA)"); // zhuzhu
    // m.def("forward_face_index_map_capture_double_side", &forward_face_index_map_capture_double_side,
    //    "FORWARD_FACE_INDEX_MAP_CAPTURE_DOUBLE_SIDE (CUDA)"); // zhuzhu
    m.def("forward_face_index_map", &forward_face_index_map, "FORWARD_FACE_INDEX_MAP (CUDA)");
    m.def("forward_texture_sampling", &forward_texture_sampling, "FORWARD_TEXTURE_SAMPLING (CUDA)");
    m.def("backward_pixel_map", &backward_pixel_map, "BACKWARD_PIXEL_MAP (CUDA)");
    m.def("backward_textures", &backward_textures, "BACKWARD_TEXTURES (CUDA)");
    m.def("backward_depth_map", &backward_depth_map, "BACKWARD_DEPTH_MAP (CUDA)");
}
