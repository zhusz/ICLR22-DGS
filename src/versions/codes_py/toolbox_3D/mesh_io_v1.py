# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from plyfile import PlyData, PlyElement
from matplotlib.cm import get_cmap
import numpy as np


def load_ply_np(fn):
    ply = PlyData.read(fn)
    vert0 = np.stack([
        ply['vertex'][k] for k in ['x', 'y', 'z']
    ], 1).astype(np.float32)
    face0 = np.stack([
        f0[0] for f0 in ply['face']
    ], 0).astype(np.int32)
    return vert0, face0


def dumpPlyPointCloud(fn, vert0, vertRgb0=None):
    assert fn.endswith('.ply')
    assert len(vert0.shape) == 2 and vert0.shape[1] == 3 and 'float' in str(vert0.dtype)
    if vertRgb0 is None:
        v0 = np.array([(vert0[j, 0], vert0[j, 1], vert0[j, 2]) for j in range(vert0.shape[0])],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        assert vertRgb0.shape == vert0.shape and 'float' in str(vertRgb0.dtype)
        assert vertRgb0.max() <= 1.00001 and vertRgb0.min() >= -0.00001
        vertRgb0UChar = (vertRgb0 * 255.).astype(np.uint8)
        v0 = np.array([(
            vert0[j, 0],
            vert0[j, 1],
            vert0[j, 2],
            vertRgb0UChar[j, 0],
            vertRgb0UChar[j, 1],
            vertRgb0UChar[j, 2],
        ) for j in range(vert0.shape[0])],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el_v0 = PlyElement.describe(v0, 'vertex')
    PlyData([el_v0]).write(fn)


def dumpPly(fn, vert0, face0, vertRgb0=None):
    assert fn.endswith('.ply')
    assert len(vert0.shape) == 2 and vert0.shape[1] == 3 and 'float' in str(vert0.dtype)
    assert len(face0.shape) == 2 and face0.shape[1] == 3 and 'int' in str(face0.dtype)
    if vertRgb0 is None:
        v0 = np.array([(vert0[j, 0], vert0[j, 1], vert0[j, 2]) for j in range(vert0.shape[0])],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        assert vertRgb0.shape == vert0.shape and 'float' in str(vertRgb0.dtype)
        assert vertRgb0.max() <= 1.00001 and vertRgb0.min() >= -0.00001
        vertRgb0UChar = (vertRgb0 * 255.).astype(np.uint8)
        v0 = np.array([(
            vert0[j, 0],
            vert0[j, 1],
            vert0[j, 2],
            vertRgb0UChar[j, 0],
            vertRgb0UChar[j, 1],
            vertRgb0UChar[j, 2],
        ) for j in range(vert0.shape[0])],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    t0 = np.array([(face0[j].tolist(), ) for j in range(face0.shape[0])],
                  dtype=[('vertex_index', 'i4', (3, ))])

    el_v0 = PlyElement.describe(v0, 'vertex')
    el_t0 = PlyElement.describe(t0, 'face')

    PlyData([el_v0, el_t0]).write(fn)


def dumpPly2(fn, vert0, face0, faceRgb0):
    assert fn.endswith('.ply')
    assert len(vert0.shape) == 2 and vert0.shape[1] == 3 and 'float' in str(vert0.dtype)
    assert len(face0.shape) == 2 and face0.shape[1] == 3 and 'int' in str(face0.dtype)

    assert faceRgb0.shape[0] == face0.shape[0] and faceRgb0.shape[1] == 3
    assert faceRgb0.max() <= 1.00001 and faceRgb0.min() >= -0.00001
    faceRgb0UChar = (faceRgb0 * 255.).astype(np.uint8)
    v0 = np.array([(
        vert0[j, 0],
        vert0[j, 1],
        vert0[j, 2],
    ) for j in range(vert0.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    t0 = np.array([(face0[j].tolist(), faceRgb0UChar[j, 0], faceRgb0UChar[j, 1], faceRgb0UChar[j, 2])
                   for j in range(face0.shape[0])],
                  dtype=[('vertex_index', 'i4', (3, )), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el_v0 = PlyElement.describe(v0, 'vertex')
    el_t0 = PlyElement.describe(t0, 'face')

    PlyData([el_v0, el_t0]).write(fn)


def dumpPly2_float(fn, vert0, face0, faceFloat0, cmapName='inferno'):
    cmap = get_cmap(cmapName)
    assert len(faceFloat0.shape) == 1 and faceFloat0.shape[0] == face0.shape[0]
    assert faceFloat0.max() <= 1
    faceRgb0 = cmap(faceFloat0.astype(np.float32))[:, :3]
    faceRgb0[faceFloat0 < 0, :] = 0
    dumpPly2(fn, vert0, face0, faceRgb0)


def load_obj_np(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    return load_obj_np_from_readlines(lines)


def load_obj_np_from_readlines(lines):
    vertices = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    vert0 = vertices
    face0 = faces
    return vert0, face0


def dump_obj_np(fn, v0, f0):
    assert fn.endswith('.obj')
    assert len(v0.shape) == 2 and v0.shape[1] == 3 and 'float' in str(v0.dtype)
    assert len(f0.shape) == 2 and f0.shape[1] == 3 and 'int' in str(f0.dtype)

    s = ""
    s += ("# OBJ file\n")
    for v in v0:
        s += ("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
    for f in f0:
        s += ("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))
    with open(fn, 'w') as fileObj:
        fileObj.write(s)
