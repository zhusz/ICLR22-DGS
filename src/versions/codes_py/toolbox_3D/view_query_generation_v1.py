# (tfconda)
import os
import numpy as np


def gen_viewport_query_simple(Lx, Ly, Lz, fScaleX, fScaleY, depthMax, zNear=1.e-6):
    # Assume the output Boundary cube is from [-depthMax / fScaleX, -depthMax / fScaleY, 0] to
    #                                         [+depthMax / fScaleX, +depthMax / fScaleY, depthMax]
    # The scale of the cube is
    #       [depthMax / fScaleX * 2 / Lx, depthMax / fScaleY * 2 / Ly, depthMax / Lz] in each dimension
    # Their boundary grid center should readjust accordingly (+0.5 * gridScale)

    # not a batch mode

    boundMin = [-depthMax / fScaleX, -depthMax / fScaleY, 0]
    boundMin = np.array(boundMin).astype(np.float32)
    boundMax = [depthMax / fScaleX, depthMax / fScaleY, depthMax]
    boundMax = np.array(boundMax).astype(np.float32)
    # gridScale = [depthMax / fScaleX * 2 / Lx, depthMax / fScaleY * 2 / Ly, depthMax / Lz]
    # gridScale = np.array(gridScale).astype(np.float32)

    return get_viewport_query_given_bound(Lx, Ly, Lz, fScaleX, fScaleY, boundMin, boundMax, zNear)


def gen_viewport_query_given_bound(Lx, Ly, Lz, fScaleX, fScaleY, boundMin, boundMax, zNear=1.e-6):
    assert len(boundMin) == 3
    assert len(boundMax) == 3
    gridScale = (boundMax - boundMin) / np.array([Lx, Ly, Lz], dtype=np.float32)

    xi = np.linspace(boundMin[0] + 0.5 * gridScale[0], boundMax[0] - 0.5 * gridScale[0], Lx).astype(np.float32)
    yi = np.linspace(boundMin[1] + 0.5 * gridScale[1], boundMax[1] - 0.5 * gridScale[1], Ly).astype(np.float32)
    zi = np.linspace(boundMin[2] + 0.5 * gridScale[2], boundMax[2] - 0.5 * gridScale[2], Lz).astype(np.float32)
    x, y, z = np.meshgrid(xi, yi, zi)

    # Note these xyz things are all in the cam sys
    xyzCam = np.stack([x, y, z], -1).reshape((-1, 3))
    xyzCamPersp = np.stack([
        fScaleX * xyzCam[:, 0] / np.clip(xyzCam[:, 2], a_min=zNear, a_max=np.inf),
        fScaleY * xyzCam[:, 1] / np.clip(xyzCam[:, 2], a_min=zNear, a_max=np.inf),
        xyzCam[:, 2],
    ], 1)
    flagQuery = (np.abs(xyzCamPersp[:, 0]) <= 1) & (np.abs(xyzCamPersp[:, 1]) <= 1) & \
                (np.abs(xyzCamPersp[:, 2]) > zNear)

    # debugging purpose
    # from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
    # v0, f0 = voxSdfSign2mesh_skmc(flagQuery.reshape((Ly, Lx, Lz)).astype(np.float32), boundMin + 0.5 * gridScale, gridScale)
    # from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
    # from codes_py.py_ext.misc_v1 import mkdir_full
    # projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    # visualDir = projRoot + 'cache/debuggingCenter/codes_py/toolbox_3D/'
    # mkdir_full(visualDir)
    # dumpPly(visualDir + 'temp.ply', v0, f0)

    extractedXyzCam = xyzCam[flagQuery, :]
    extractedXyzCamPersp = xyzCamPersp[flagQuery, :]

    # xyzCam[flagQuery, :] = extractedXyzCam  # Yes this is viable

    # Put everything into the package
    viewQueryPackage = {
        'extractedXyzCam': extractedXyzCam,
        'extractedXyzCamPersp': extractedXyzCamPersp,
        'xyzCam': xyzCam,
        'xyzCamPersp': xyzCamPersp,
        'flagQuery': flagQuery,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'fScaleX': fScaleX, 'fScaleY': fScaleY,
        'zNear': zNear,
        'boundMin': boundMin,
        'boundMax': boundMax,
        'gridScale': gridScale,
        'goxyzCam': boundMin + gridScale * 0.5,
        'sCell': gridScale,
    }

    # import ipdb
    # ipdb.set_trace()
    # print(1 + 1)

    return viewQueryPackage


if __name__ == '__main__':
    gen_viewport_query_simple(100, 100, 50, 1, 1, 10)

