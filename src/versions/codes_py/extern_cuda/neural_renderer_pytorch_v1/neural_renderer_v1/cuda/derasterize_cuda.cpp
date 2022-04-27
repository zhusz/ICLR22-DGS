#include <torch/torch.h>

#include <vector>

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

#define MAX_FLOAT 999999999.9
#define SQR(x) ((x) * (x))

// C++ interface

#define CHECK_NOT_CUDA(x) TORCH_CHECK(!x.type().is_cuda(), #x "must not be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_NOT_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void assign_pixels_into_triangle_ids(
        const int* _triangle_id,
        vector<vector<int> >& I,
        vector<vector<int> >& J,
        int image_height,
        int image_width,
        int n_faces) {

    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            const int f = _triangle_id[i * image_width + j];
            if ((f >= 0) and (f < n_faces)) {
                I[f].push_back(i);
                J[f].push_back(j);
            }
        }
    }
}

void solve_face_sixes(
        const float* _bary,
        const vector<int>& If,
        const vector<int>& Jf,
        float* _face_verts_2D_f,
        unsigned char* _flag_faces_f,
        int image_height,
        int image_width,
        float over_determined_factor,
        bool pixel_is_area,
        float valid_dist_thre_in_pixels) {

    const int n_pixels_f = If.size();
    assert(n_pixels_f == Jf.size());
    if (n_pixels_f * 2 < 6 * over_determined_factor) {
        return;
    }
    MatrixXf A(n_pixels_f * 2, 6);
    MatrixXf b(n_pixels_f * 2, 1);
    for (int k = 0; k < n_pixels_f; k++) {
        const int i = If[k];
        const int j = Jf[k];
        const float bc0 = _bary[(i * image_width + j) * 3 + 0];
        const float bc1 = _bary[(i * image_width + j) * 3 + 1];
        const float bc2 = _bary[(i * image_width + j) * 3 + 2];
        float xc, yc;
        if (!pixel_is_area) {
            yc = (i + .5) / image_height * 2. - 1.;
            xc = (j + .5) / image_width * 2. - 1.;
        } else {
            yc = (float)i / image_height * 2. - 1.;
            xc = (float)j / image_width * 2. - 1.;
        }
        A(2 * k + 0, 0) = bc0;
        A(2 * k + 0, 1) = 0;
        A(2 * k + 0, 2) = bc1;
        A(2 * k + 0, 3) = 0;
        A(2 * k + 0, 4) = bc2;
        A(2 * k + 0, 5) = 0;
        A(2 * k + 1, 0) = 0;
        A(2 * k + 1, 1) = bc0;
        A(2 * k + 1, 2) = 0;
        A(2 * k + 1, 3) = bc1;
        A(2 * k + 1, 4) = 0;
        A(2 * k + 1, 5) = bc2;
        b(2 * k + 0) = xc;
        b(2 * k + 1) = yc;
    }
    MatrixXf six = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    bool flag_triangle = 1;
    for (int z = 0; z < 3; z++) {
        bool flag_z = 0;
        for (int k = 0; k < n_pixels_f; k++) {
            const int i = If[k];
            const int j = Jf[k];
            float xc, yc;
            if (!pixel_is_area) {
                yc = (i + .5);
                xc = (j + .5);
            } else {
                yc = (float)i;
                xc = (float)j;
            }
            const float dist_y_2 = SQR((six(2 * z + 1) + 1) / 2. * image_height - yc);
            const float dist_x_2 = SQR((six(2 * z + 0) + 1) / 2. * image_width - xc);
            const float dist2 = dist_y_2 + dist_x_2;
            if (dist2 < SQR(valid_dist_thre_in_pixels)) {
                flag_z = 1;
                break;
            }
        }
        if (!flag_z) {
            flag_triangle = 0;
            break;
        }
    }
    *_flag_faces_f = (unsigned char) flag_triangle;
    if (flag_triangle) {
        for (int z = 0; z < 6; z++) {
            _face_verts_2D_f[z] = six(z);
        }
    }
}

vector<at::Tensor> FBC2faceVerts2D(
        at::Tensor triangle_id, // batch_size * image_height * image_width
        at::Tensor bary, // batch_size * image_height * image_width * 3
        at::Tensor face_verts_2D, // batch_size * n_faces * 3(v012) * 2(xy)
        at::Tensor flag_faces, // batch_size * n_faces
        float over_determined_factor,
        bool pixel_is_area,
        float valid_dist_thre_in_pixels) {

    CHECK_INPUT_CPU(triangle_id);
    CHECK_INPUT_CPU(bary);
    CHECK_INPUT_CPU(face_verts_2D);
    CHECK_INPUT_CPU(flag_faces);

    const int batch_size = triangle_id.size(0);
    const int image_height = triangle_id.size(1);
    const int image_width = triangle_id.size(2);
    const int n_faces = face_verts_2D.size(1);

    for (int b = 0; b < batch_size; b++) {
        const int* _triangle_id = triangle_id.data<int>() + b * image_height * image_width;
        const float* _bary = bary.data<float>() + b * image_height * image_width * 3;
        float* _face_verts_2D = face_verts_2D.data<float>() + b * n_faces * 3 * 2;
        unsigned char* _flag_faces = flag_faces.data<unsigned char>() + b * n_faces;

        // collect all the pixels into different triangle_ids
        vector<vector<int> > I, J;
        I.resize(n_faces);
        J.resize(n_faces);
        assign_pixels_into_triangle_ids(_triangle_id, I, J, image_height, image_width, n_faces);

        // solve the six xys for every face
        for (int f = 0; f < n_faces; f++) {
            solve_face_sixes(_bary, I[f], J[f], _face_verts_2D + f * 3 * 2, _flag_faces + f, image_height, image_width,
                             over_determined_factor, pixel_is_area, valid_dist_thre_in_pixels);
        }
    }

    return {face_verts_2D, flag_faces};
}

vector<at::Tensor> faceTripleInfo2vertInfo(
        at::Tensor faceInfo, // batch_size * n_faces * 3(v012) * dim_info
        at::Tensor flag_faces, // batch_size * n_faces
        at::Tensor tris, // batch_size * n_faces * 3(v012)
        at::Tensor vertInfo, // batch_size * n_verts * dim_info
        at::Tensor flag_verts // batch_size * n_verts
        ){

    CHECK_INPUT_CPU(faceInfo);
    CHECK_INPUT_CPU(flag_faces);
    CHECK_INPUT_CPU(tris);
    CHECK_INPUT_CPU(vertInfo);
    CHECK_INPUT_CPU(flag_verts);

    const int batch_size = faceInfo.size(0);
    const int n_faces = faceInfo.size(1);
    const int n_verts = vertInfo.size(1);
    const int dim_info = faceInfo.size(3);

    for (int b = 0; b < batch_size; b++) {
        const float* _faceInfo = faceInfo.data<float>() + b * n_faces * 3 * dim_info;
        const unsigned char* _flag_faces = flag_faces.data<unsigned char>() + b * n_faces;
        const int* _tris = tris.data<int>() + b * n_faces * 3;
        float* _vertInfo = vertInfo.data<float>() + b * n_verts * dim_info;
        unsigned char* _flag_verts = flag_verts.data<unsigned char>() + b * n_verts;

        vector<int> vert_count(n_verts, 0);
        for (int f = 0; f < n_faces; f++) {
            if (_flag_faces[f] == 1) { // only process the valid faces
                for (int j = 0; j < 3; j++) {
                    const int v = _tris[f * 3 + j];
                    vert_count[v]++;
                    for (int d = 0; d < dim_info; d++) {
                        _vertInfo[v * dim_info + d] += _faceInfo[(f * 3 + j) * dim_info + d];
                    }
                }
            }
        }
        for (int v = 0; v < n_verts; v++) {
            if (vert_count[v] > 0) {
                _flag_verts[v] = 1;
            } else {
                _flag_verts[v] = 0;
            }
            for (int d = 0; d < dim_info; d++) {
                _vertInfo[v * dim_info + d] /= (float)vert_count[v];
            }
        }
    }

    return {vertInfo, flag_verts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("FBC2faceVerts2D", &FBC2faceVerts2D, "FBC2faceVerts2D (CPU)");
    m.def("faceTripleInfo2vertInfo", &faceTripleInfo2vertInfo, "faceTripleInfo2vertInfo (CPU)");
}

