#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations

std::vector<at::Tensor> sdf_cuda(
        at::Tensor queryPoint,
        at::Tensor vert,
        at::Tensor face,
        at::Tensor referencePoints,
        at::Tensor sdf,
        at::Tensor projPoint,
        at::Tensor cudaIntBuffer);

std::vector<at::Tensor> udf_cuda(
        at::Tensor queryPoint,
        at::Tensor vert,
        at::Tensor face,
        at::Tensor udf,
        at::Tensor projPoint);

std::vector<at::Tensor> sign_value_matching_cuda(
        at::Tensor mismatchedSdf,
        int radius,
        at::Tensor matchedSdf
);

std::vector<at::Tensor> sdf(
        at::Tensor queryPoint,
        at::Tensor vert,
        at::Tensor face,
        at::Tensor referencePoints,
        at::Tensor sdf,
        at::Tensor projPoint,
        at::Tensor cudaIntBuffer) {

	CHECK_INPUT(queryPoint);
	CHECK_INPUT(vert);
	CHECK_INPUT(face);
	CHECK_INPUT(referencePoints);
	CHECK_INPUT(sdf);
	CHECK_INPUT(projPoint);
	CHECK_INPUT(cudaIntBuffer);

	return sdf_cuda(
	    queryPoint,
	    vert, face,
	    referencePoints,
	    sdf, projPoint, cudaIntBuffer);
}

std::vector<at::Tensor> udf(  // no triangle hashing, naive
        at::Tensor queryPoint,
        at::Tensor vert,
        at::Tensor face,
        at::Tensor udf,
        at::Tensor projPoint) {

    CHECK_INPUT(queryPoint);
	CHECK_INPUT(vert);
	CHECK_INPUT(face);
	CHECK_INPUT(udf);
	CHECK_INPUT(projPoint);

	return udf_cuda(
	    queryPoint,
	    vert, face,
	    udf, projPoint);
}

std::vector<at::Tensor> sign_value_matching(
        at::Tensor mismatchedSdf,
        int radius,
        at::Tensor matchedSdf
) {
    CHECK_INPUT(mismatchedSdf);
    CHECK_INPUT(matchedSdf);
    return sign_value_matching_cuda(
        mismatchedSdf, radius, matchedSdf
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdf", &sdf, "SDF (CUDA)");
    m.def("udf", &udf, "UDF (CUDA)");
    m.def("sign_value_matching", &sign_value_matching,
        "SIGN VALUE MATCHING (CUDA)");
}
