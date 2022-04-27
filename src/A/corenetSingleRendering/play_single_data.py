# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from scene import load_from_npz
import matplotlib.pyplot as plt
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np, dumpPly
import numpy as np
from codes_py.py_ext.misc_v1 import mkdir_full


def main():
    # set your set
    dataset = 'corenetSingleRendering'
    sptTag = 'single'  # single / pairs / triplets
    splitTag = 'train'  # train / val / test
    corenetShaH3Tag = 'fff'
    corenetShaTag = 'fffffdadef3f7a7b2101c8312adee246db46590340fbce307647ce2c245057c1'

    # meta
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    visualDir = projRoot + 'cache/dataset/%s/play_single_data/' % dataset
    mkdir_full(visualDir)

    # npz
    npz_path = projRoot + 'remote_fastdata/corenet/data/%s.%s/%s/%s.npz' % (
            sptTag, splitTag, corenetShaH3Tag, corenetShaTag)
    result = load_from_npz(
        path=npz_path,
        meshes_dir=projRoot + 'remote_fastdata/corenet/data/shapenet_meshes/',
        load_extra_fields=False,
    )
    d = dict(np.load(npz_path))
    catID = str(d['mesh_labels'][0])
    shaID = str(d['mesh_filenames'][0])
    m = dict(np.load(projRoot + 'remote_fastdata/corenet/data/shapenet_meshes/%s/%s.npz' % (catID, shaID)))

    import ipdb
    ipdb.set_trace()

    # mesh original
    vert0, face0 = load_obj_np(
        projRoot + 'remote_fastdata/shapeNetV2/ShapeNetCore.v2/%s/%s/models/model_normalized.obj' % (
            catID, shaID,
        ))
    dumpPly(
        visualDir + 'rawMesh_%s_%s_%s_%s.ply' % (sptTag, splitTag, corenetShaH3Tag, corenetShaTag),
        vert0, face0,
    )

    plt.imshow(result.pbrt_image)

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
