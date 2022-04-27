#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

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
        int L);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> IUV2FBC(
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

    CHECK_INPUT(I);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(All_U_norm);
    CHECK_INPUT(All_V_norm);
    CHECK_INPUT(All_FaceIndices);
    CHECK_INPUT(All_Faces);
    CHECK_INPUT(F);
    CHECK_INPUT(BC);

    return IUV2FBC_cuda(I,
                        U,
                        V,
                        All_U_norm,
                        All_V_norm,
                        All_FaceIndices,
                        All_Faces,
                        F,
                        BC,
                        flag_inside,
                        L);
    return {F, BC, flag_inside};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("IUV2FBC", &IUV2FBC, "IUV2FBC (CUDA)");
}