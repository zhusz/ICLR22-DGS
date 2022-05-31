#include <torch/torch.h>
#include <vector>

at::Tensor faceVert_find_next_cuda(at::Tensor face,
                               at::Tensor validFaceFlag,
                               at::Tensor nextFaceVertLoc);

at::Tensor count_cycle_face_indexing_cuda(at::Tensor nextFaceIndexingLoc,
                                          at::Tensor validFaceFlag,
                                          at::Tensor sumFaceIndexingLoc);

at::Tensor faceEdge_find_next_cuda(at::Tensor face,
                              at::Tensor validFaceFlag,
                              at::Tensor nextFaceVertLoc,
                              at::Tensor sumFaceVert,
                              at::Tensor nextFaceEdgeLoc);

at::Tensor calc_face_edge_normal_cuda(at::Tensor faceNormal,
                                      at::Tensor validFaceFlag,
                                      at::Tensor nextFaceEdgeLoc,
                                      at::Tensor sumFaceEdge,
                                      at::Tensor faceEdgeNormal);

at::Tensor faceVert_find_next(at::Tensor face,
                              at::Tensor validFaceFlag,
                              at::Tensor nextFaceVertLoc) {

    return faceVert_find_next_cuda(face,
                                   validFaceFlag,
                                   nextFaceVertLoc);
}

at::Tensor faceEdge_find_next(at::Tensor face,
                                   at::Tensor validFaceFlag,
                                   at::Tensor nextFaceVertLoc,
                                   at::Tensor sumFaceVert,
                                   at::Tensor nextFaceEdgeLoc) {

    return faceEdge_find_next_cuda(face,
                                   validFaceFlag,
                                   nextFaceVertLoc,
                                   sumFaceVert,
                                   nextFaceEdgeLoc);
}

at::Tensor count_cycle_face_indexing(at::Tensor nextFaceIndexingLoc,
                                     at::Tensor validFaceFlag,
                                     at::Tensor sumFaceIndexingLoc) {

    return count_cycle_face_indexing_cuda(nextFaceIndexingLoc,
                                   validFaceFlag,
                                   sumFaceIndexingLoc);
}

at::Tensor calc_face_edge_normal(at::Tensor faceNormal,
                                      at::Tensor validFaceFlag,
                                      at::Tensor nextFaceEdgeLoc,
                                      at::Tensor sumFaceEdge,
                                      at::Tensor faceEdgeNormal) {

    return calc_face_edge_normal_cuda(faceNormal,
                                      validFaceFlag,
                                      nextFaceEdgeLoc,
                                      sumFaceEdge,
                                      faceEdgeNormal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("faceVert_find_next", &faceVert_find_next, "faceVert_find_next (GPU)");
    m.def("faceEdge_find_next", &faceEdge_find_next, "faceEdge_find_next (GPU)");
    m.def("count_cycle_face_indexing", &count_cycle_face_indexing, "count_cycle_face_indexing (GPU)");
    m.def("calc_face_edge_normal", &calc_face_edge_normal, "calc_face_edge_normal (GPU)");
}

