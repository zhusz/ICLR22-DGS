# Note this function need to import pymesh, which needs efforts to install (not major efforst but still efforts)
# Hence, import this file only if necessary
import pymesh
from .mesh_v1 import vertInfo2faceVertInfoNP
from .mesh_surgery_v1 import trimVert, combineMultiShapes_plain
import numpy as np


def pymesh_mesh_subdivion_wrapper(vert0, face0, order):
    mesh = pymesh.form_mesh(vert0, face0)
    mesh = pymesh.subdivide(mesh, order=order, method='simple')
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def selectiveMeshSubdivision(vert0, face0, order, edgeLengthThre, verbose):
    # Note because the subdivision process devide mesh into parts
    # the resulting mesh might not be manifold
    counter = 0
    collector = []
    while True:
        faceVert0 = vertInfo2faceVertInfoNP(vert0[None], face0[None])[0]
        edgeLength0 = np.stack([
            ((faceVert0[:, 1, :] - faceVert0[:, 2, :]) ** 2).sum(1) ** 0.5,
            ((faceVert0[:, 2, :] - faceVert0[:, 0, :]) ** 2).sum(1) ** 0.5,
            ((faceVert0[:, 0, :] - faceVert0[:, 1, :]) ** 2).sum(1) ** 0.5,
        ], 1)
        edgeLengthMax0 = edgeLength0.max(1)
        if verbose > 0 and counter % verbose == 0:
            print('    selectiveMeshSubdivision: counter %d, edgeLengthMax %.3f' %
                  (counter, edgeLengthMax0.max()))
        mask0 = (edgeLengthMax0 > edgeLengthThre)
        v0, f0 = trimVert(vert0, face0[mask0 == 0])
        collector.append((v0, f0))
        if mask0.sum() == 0:
            break
        else:
            vert0, face0 = pymesh_mesh_subdivion_wrapper(vert0, face0[mask0], order)
            vert0, face0 = trimVert(vert0, face0)

            counter += 1

    vert0, face0 = combineMultiShapes_plain(collector)
    vert0, face0 = trimVert(vert0, face0)
    return vert0, face0
