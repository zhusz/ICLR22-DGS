#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> dgs_forward_cuda(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi_thgpu,
    at::Tensor phi_on_i_thgpu,
    at::Tensor phi_on_j_thgpu,
    at::Tensor debugging_info
);

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
);

std::vector<at::Tensor> dgs3d_forward_cuda(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi4,
    at::Tensor debugging_info
);

std::vector<at::Tensor> dgs3d_backward_cuda(
    at::Tensor grad_phi4,
    at::Tensor pCam,
    at::Tensor grad_input,
    at::Tensor debugging_info
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> dgs_forward(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi,
    at::Tensor phi_on_i,
    at::Tensor phi_on_j,
    at::Tensor debugging_info
) {
    CHECK_INPUT(input);
    CHECK_INPUT(grid);
    CHECK_INPUT(phi);
    CHECK_INPUT(phi_on_i);
    CHECK_INPUT(phi_on_j);
    CHECK_INPUT(debugging_info);
    return dgs_forward_cuda(
        input, grid, phi, phi_on_i, phi_on_j, debugging_info
    );
}

std::vector<at::Tensor> dgs_backward(
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
    CHECK_INPUT(partialL_over_phi);
    CHECK_INPUT(partialL_over_phiOverXyzCam);
    CHECK_INPUT(grid);
    CHECK_INPUT(fh_over_z);
    CHECK_INPUT(fw_over_z);
    CHECK_INPUT(yCamPerspQueryMap_over_z);
    CHECK_INPUT(xCamPerspQueryMap_over_z);
    CHECK_INPUT(partialL_over_feat);
    CHECK_INPUT(debugging_info);
    return dgs_backward_cuda(
        partialL_over_phi,
        partialL_over_phiOverXyzCam,
        grid,
        fh_over_z,
        fw_over_z,
        yCamPerspQueryMap_over_z,
        xCamPerspQueryMap_over_z,
        partialL_over_feat,
        debugging_info
    );
}

std::vector<at::Tensor> dgs3d_forward(
    at::Tensor input,
    at::Tensor grid,
    at::Tensor phi4,
    at::Tensor debugging_info
) {
    CHECK_INPUT(input);
    CHECK_INPUT(grid);
    CHECK_INPUT(phi4);
    CHECK_INPUT(debugging_info);
    return dgs3d_forward_cuda(
        input,
        grid,
        phi4,
        debugging_info
    );
}

std::vector<at::Tensor> dgs3d_backward(
    at::Tensor grad_phi4,
    at::Tensor pCam,
    at::Tensor grad_input,
    at::Tensor debugging_info
) {
    CHECK_INPUT(grad_phi4);
    CHECK_INPUT(pCam);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(debugging_info);
    return dgs3d_backward_cuda(
        grad_phi4,
        pCam,
        grad_input,
        debugging_info
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dgs_forward", &dgs_forward, "DGS_FORWARD (CUDA)");
    m.def("dgs_backward", &dgs_backward, "DGS_BACKWARD (CUDA)");
    m.def("dgs3d_forward", &dgs3d_forward, "DGS3D_FORWARD (CUDA)");
    m.def("dgs3d_backward", &dgs3d_backward, "DGS3D_BACKWARD (CUDA)");
}
