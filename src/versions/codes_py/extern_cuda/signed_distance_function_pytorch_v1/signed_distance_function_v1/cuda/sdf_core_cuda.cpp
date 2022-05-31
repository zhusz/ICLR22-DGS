#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> point_id_rearrange_cpu(
        at::Tensor pointGridXYZ,
        at::Tensor pointIDOrderingBuffer,
        at::Tensor gridPointStartingIndex);

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
        float aGrid);


std::vector<at::Tensor> point_id_rearrange(
        at::Tensor pointGridXYZ,
        at::Tensor pointIDOrderingBuffer,
        at::Tensor gridPointStartingIndex) {

    return point_id_rearrange_cpu(pointGridXYZ, pointIDOrderingBuffer, gridPointStartingIndex);
}

std::vector<at::Tensor> sdf_tris_search(
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

    return sdf_tris_search_cuda(queryPoints,
                                trisIDOrderingBuffer,
                                gridTrisStartingIndex,
                                faceVert,
                                faceNormal,
                                faceEdgeNormal,
                                faceVertNormal,
                                validFaceFlag,
                                sdfPos,
                                sdfNeg,
                                projPointPos,
                                projPointNeg,
                                projPointNormalPos,
                                projPointNormalNeg,
                                projTrisIDPos,
                                projTrisIDNeg,
                                debuggingInfoExport,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                zmin,
                                zmax,
                                aGrid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("point_id_rearrange", &point_id_rearrange, "point_id_rearrange (CUDA)");
    m.def("sdf_tris_search", &sdf_tris_search, "sdf_tris_search (CUDA)");
}

