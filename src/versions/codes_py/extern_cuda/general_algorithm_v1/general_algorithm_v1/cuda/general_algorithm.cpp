#include <torch/torch.h>

#include <vector>

at::Tensor batched_binary_search_over_cumsumed_vec_cuda(
    at::Tensor values,
    at::Tensor cumsumed_vec,
    at::Tensor returned_index
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor batched_binary_search_over_cumsumed_vec(
    at::Tensor values,
    at::Tensor cumsumed_vec,
    at::Tensor returned_index
) {
    CHECK_INPUT(values);
    CHECK_INPUT(cumsumed_vec);
    CHECK_INPUT(returned_index);
    return batched_binary_search_over_cumsumed_vec_cuda(
        values,
        cumsumed_vec,
        returned_index
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_binary_search_over_cumsumed_vec",
        &batched_binary_search_over_cumsumed_vec, "BATCHED_BINARY_SEARCH_OVER_CUMSUMED_VECJK (CUDA)");
}
